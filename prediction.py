import json
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
intents = json.loads(open("intents.json").read())

model = load_model("chatbot_model.keras")

def preprocess_input(input):
    tokenized_sentence = word_tokenize(input)
    lem_tokens = [lemmatizer.lemmatize(word) for word in tokenized_sentence]
    bag = []
    for word in words:
        bag.append(1) if word in lem_tokens else bag.append(0)
    print("bag : ", len(bag))
    return bag
    
def predict(input):
    processed_input = preprocess_input(input)
    processed_input = np.expand_dims(processed_input, axis=0)
    print(processed_input.shape)
    output = model.predict(processed_input)        
    pred_class = classes[np.argmax(output[0])]
    return pred_class

def response(input):
    pred_class = predict(input)
    responses = [intent['responses'] for intent in intents["intents"] if intent['tag'] == pred_class]
    return random.choice(responses[0])
    
while True:
    q = input("enter input here : ")
    res = response(q)
    print("response : ", res )
    