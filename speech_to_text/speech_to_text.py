import speech_recognition as sr

class SpeechToText:
    def __init__(self, audio_path):
        self.audio_path = audio_path

    def convert(self, api_key=None):
        try:
            r = sr.Recognizer()
            raw_audio = sr.AudioFile(self.audio_path)

            with raw_audio as source:
                audio = r.record(source)
                
            if api_key:
                return r.recognize_google(audio, key=api_key)
            else:
                return r.recognize_google(audio)
        except Exception as e:
            print(f'There was an error while converting from text to speech: {e}')
