import collections
import numpy as np
from transformers import AutoTokenizer,TFAutoModelForQuestionAnswering

class DocumentReader:
    def __init__(self, model_name, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(model_name, from_pt=True)

    def get_answer(self, question, context, nbest=5, null_thresh=-1.1585063934326172):
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_special_tokens_mask=True, return_tensors='tf')
        start_scores, end_scores = self.model(inputs)

        start_scores_list = start_scores.numpy()[0]
        end_scores_list = end_scores.numpy()[0]

        start_index_and_score = sorted(enumerate(start_scores_list), key=lambda x: x[1], reverse=True)
        end_index_and_score = sorted(enumerate(end_scores_list), key=lambda x: x[1], reverse=True)

        start_indexes = [idx for idx, score in start_index_and_score[:5]]
        end_indexes = [idx for idx, score in end_index_and_score[:5]]

        preliminary_prediction = collections.namedtuple("preliminary_prediction", ["start_index", "end_index", "start_score", "end_score"])

        tokens = list(inputs['input_ids'][0].numpy())

        question_indexes = [i+1 for i, token in enumerate(tokens[1:tokens.index(3)])]

        preliminary_preds = []

        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index in question_indexes:
                    continue

                if end_index in question_indexes:
                    continue

                if end_index < start_index:
                    continue

                preliminary_preds.append(preliminary_prediction(start_index=start_index,
                                                                end_index=end_index,
                                                                start_score=start_scores.numpy()[0][start_index],
                                                                end_score=end_scores.numpy()[0][end_index]))

        prelim_preds = sorted(preliminary_preds, key=lambda x: (x.start_score + x.end_score), reverse=True)

        best_prediction = collections.namedtuple("best_prediction", ["text", "start_score", "end_score"])

        nbest = []
        seen_predictions = []

        for pred in prelim_preds:
            if len(nbest) >= 5:
                break

            if pred.start_index > 0:
                text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(tokens[pred.start_index:pred.end_index+1]))
                text = text.strip()
                text = " ".join(text.split())
            else:
                continue

            if text in seen_predictions:
                continue

            seen_predictions.append(text)
            nbest.append(best_prediction(text=text, start_score=pred.start_score, end_score=pred.end_score))

        nbest.append(best_prediction(text="", start_score=start_scores.numpy()[0][0], end_score=end_scores.numpy()[0][0]))

        probabilities = self._prediction_probabilities(nbest)

        null_score = start_scores.numpy()[0][0] + end_scores.numpy()[0][0]

        score_diff = null_score - nbest[0].start_score - nbest[0].end_score

        if score_diff > null_thresh:
            return "", probabilities[-1]
        else:
            return  nbest[0].text, probabilities[0]

    def _prediction_probabilities(self, predictions):
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        all_scores = [pred.start_score+pred.end_score for pred in predictions]
        return softmax(np.array(all_scores))
