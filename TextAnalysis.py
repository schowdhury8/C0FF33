import sys
import speech_recognition as sr
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from keras.utils.np_utils import to_categorical
import numpy as np

def get_text(voice_file):
    try:
        r = sr.Recognizer()
        audio = sr.AudioFile(voice_file)
        with audio as source:
            audio_data = r.record(source)
        audio_str = r.recognize_google(audio_data)
        return audio_str
    except sr.UnknownValueError:
        print ('Google Speech Recognizer could not undersand audio.')
    except sr.RequestError:
        print ('Could not request results from Google Speech Recognition service')

def clean_text(text):
    stop_words = ["i", "me", "my","also", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",  "between", "into", "through", "during","before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "under", "again", "further", "then", "once", "here", "there", "when", "where", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such","only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    tokens = word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    clean_str = ' '.join(words)
    return clean_str
    
def main():
    audio_file = sys.argv[1]
    audio_str = get_text(audio_file)
    print('=======================')
    print(audio_str)
    clean_str = clean_text(audio_str)
    print('=======================')
    print(clean_str)
    clean_str = [clean_str,'']
    max_features = 100
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(clean_str)
    X = tokenizer.texts_to_sequences(clean_str)

    model_posneg = load_model('./text_models/binary_model.h5')
    model_multi = load_model('./text_models/multi_model.h5')
 
    X = pad_sequences(X, maxlen=57)

    prediction_multi = model_multi.predict(np.array(X))

    X = pad_sequences(X, maxlen=20)

    prediction_posneg = model_posneg.predict(np.array(X))

    print(prediction_posneg[0])
    print(prediction_multi[0])
    
    out = open("out.csv",'w')
    out.write('negative,positive\n')
    out.write(','.join(map(str,prediction_posneg[0]))+'\n')
    out.write('anger,disgust,fear,guilt,joy,sadness,shame\n')
    out.write(','.join(map(str,prediction_multi[0])))
    out.close()

main()


