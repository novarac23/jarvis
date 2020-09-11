
import numpy as np
import tensorflow as tf

from .td_utils import *
from pydub import AudioSegment
from keras import backend as K

class TriggerWordDetection:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path,
                                                custom_objects={'f1_score': self._f1_score})

    def detect_trigger_word(self, audio_path):
        self._pre_process_audio(audio_path)
        x_test = graph_spectrogram(audio_path)
        x_test = x_test.swapaxes(0,1)
        x_test = np.expand_dims(x_test, axis=0)
        predictions = self.model.predict(x_test)
        return predictions

    def cut_audio_on_trigger_word(self, filename, predictions, threshold):
        audio_clip = AudioSegment.from_wav(filename)
        t_y = predictions.shape[1]

        for i in range(t_y):
            if predictions[0,i,0] > threshold:
                start_time = ((i / t_y) * audio_clip.duration_seconds)*1000
                cut_audio = audio_clip[start_time:]

        out_file = f'cut_{filename}'
        cut_audio.export(out_file, format='wav')
        return out_file

    def _f1_score(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    def _pre_process_audio(self, file_path):
        silent_audio = AudioSegment.silent(duration=10000)
        background = AudioSegment.from_wav(file_path)
        background = silent_audio.overlay(background)
        background = background.set_frame_rate(44100)
        background.export(file_path, format="wav")
