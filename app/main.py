import os
import json
import tensorflow as tf

from flask import jsonify
from pydub import AudioSegment
from flask import Flask, render_template, request

from q_a_model.document_reader import DocumentReader
from q_a_model.paragraph_ranker import ParagraphRanker
from speech_to_text.speech_to_text import SpeechToText
from q_a_model.document_retriever import DocumentRetriever
from trigger_word_model.trigger_word_detection import TriggerWordDetection







#question = "When was Michael Jordan born?" # got it right w/ title=True and Title=False
#question = "what is population of New York city?" # got it right w/ Title=True and Title=False
#question = "Who was the first president of United States?" # got it right w/ Title=True and Title=False
#question = "How tall is Eiffel Tower in Paris?" # got it right w/ title=True
#question = "How many people live in Columbus Ohio?" # failed to answer with both Title=True and Title=False
#question = "What is the population of Columbus Ohio?" # worked with Title=True
# question = "When was Novak Djokovic born?" # worked with Title=True
# question = "When did World War II start?" # worked with Title=True
# question = "When did French revolution begin?" # worked with title=False
# question = "When did French revolution start?" # worked with Title=False

app = Flask(__name__)

ES_INDEX = os.environ.get('ES_INDEX')
ES_CONFIG = {'host': os.environ.get('ES_HOST'), 'port': os.environ.get('ES_PORT')}
MODEL_NAME = os.environ.get('MODEL_NAME')
TOKENIZER_NAME = os.environ.get('TOKENIZER_NAME')
TRIGGER_WORD_MODEL_PATH = os.environ.get('TRIGGER_WORD_MODEL_PATH')

@app.route('/get_answer', methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        raW_audio_file_name = 'file7.wav'

        f = open(raW_audio_file_name, 'wb')
        request.files['audio_data'].save(f)
        f.close()

        silent_audio = AudioSegment.silent(duration=10000)
        background = AudioSegment.from_wav(raW_audio_file_name)
        background = silent_audio.overlay(background)
        background = background.set_frame_rate(44100)
        background.export(raW_audio_file_name, format="wav")

        twd = TriggerWordDetection(TRIGGER_WORD_MODEL_PATH)
        predictions = twd.detect_trigger_word(raW_audio_file_name)
        audio_path = twd.cut_audio_on_trigger_word(raW_audio_file_name, predictions, .95)

        ind = tf.argmax(predictions, axis=1)
        print(predictions[0, int(ind[0][0]), 0])

        stt = SpeechToText(audio_path)
        question = stt.convert()
        question += "?"
        print(question)

        dr = DocumentRetriever(ES_CONFIG)
        documents = dr.retrieve_docs(question, ES_INDEX, size=5, title=False)

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

        return jsonify(str(results))
    else:
        return render_template("index.html")

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
