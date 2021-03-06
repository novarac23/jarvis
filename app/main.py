import os
import json
import random
import tensorflow as tf

from flask import jsonify
from pydub import AudioSegment
from flask import Flask, render_template, request

from q_a_model.document_reader import DocumentReader
from q_a_model.paragraph_ranker import ParagraphRanker
from speech_to_text.speech_to_text import SpeechToText
from q_a_model.document_retriever import DocumentRetriever
from trigger_word_model.trigger_word_detection import TriggerWordDetection

app = Flask(__name__)

ES_INDEX = os.environ.get('ES_INDEX')
ES_CONFIG = {'host': os.environ.get('ES_HOST'), 'port': os.environ.get('ES_PORT')}
MODEL_NAME = os.environ.get('MODEL_NAME')
TOKENIZER_NAME = os.environ.get('TOKENIZER_NAME')
TRIGGER_WORD_MODEL_PATH = os.environ.get('TRIGGER_WORD_MODEL_PATH')

@app.route('/get_answer', methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        raw_audio_file_name = f"{random.randint(1, 100000)}.wav"

        f = open(raw_audio_file_name, 'wb')
        request.files['audio_data'].save(f)
        f.close()
        
        _pre_process_audio(raw_audio_file_name)
        results = _get_answer(raw_audio_file_name)
        best_answer = _get_best_answer(results)

        if best_answer[0] == '':
            results = _get_answer(raw_audio_file_name, title=False)
            best_answer = _get_best_answer(results)

        return jsonify(str(best_answer))
    else:
        return render_template("index.html")

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _get_best_answer(results):
    print(results)
    non_empty_answers = []

    for result in results:
        if result[0] != '':
            non_empty_answers.append(result)

    if len(non_empty_answers) > 0:
        non_empty_answers.sort(key=lambda x: x[1], reverse=True)
        return non_empty_answers[0]
    else:
        results.sort(key=lambda x: x[1], reverse=True) #sort by key
        return results[0]

def _pre_process_audio(raw_audio_file_name):
    silent_audio = AudioSegment.silent(duration=10000)
    background = AudioSegment.from_wav(raw_audio_file_name)
    background = silent_audio.overlay(background)
    background = background.set_frame_rate(44100)
    background.export(raw_audio_file_name, format="wav")

def _get_answer(raw_audio_file_name, title=True):
    twd = TriggerWordDetection(TRIGGER_WORD_MODEL_PATH)
    predictions = twd.detect_trigger_word(raw_audio_file_name)
    audio_path = twd.cut_audio_on_trigger_word(raw_audio_file_name, predictions, .95)

    ind = tf.argmax(predictions, axis=1)
    print(predictions[0, int(ind[0][0]), 0])

    stt = SpeechToText(audio_path)
    question = stt.convert()
    question += "?"
    question = question.lower()
    print(question)

    dr = DocumentRetriever(ES_CONFIG)
    documents = dr.retrieve_docs(question, ES_INDEX, size=5, title=title)

    pr = ParagraphRanker(documents)
    contexts = pr.rank_paragaraphs(question, n_paragraphs=8)

    doc_reader = DocumentReader(MODEL_NAME, TOKENIZER_NAME)

    results = []

    for i, context in enumerate(contexts):
        try:
            if len(context) > 512:
                chunks = list(get_chunks(context, 511))
                for chunk in chunks:
                    final_answer = doc_reader.get_answer(question, chunk)
                    results.append(final_answer)


            final_answer = doc_reader.get_answer(question, context)

            results.append(final_answer)
        except Exception as e:
            print(f'We could not process item under {i}. Reason is: {e}')

    return results

