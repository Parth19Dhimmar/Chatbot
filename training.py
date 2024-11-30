import random
import json
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, InputLayer
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = [] # to get all the words in pattern
classes = [] # to get tag of the pattern
documents = [] # list of (tokenized word pattern list, tag name related to that pattern)
ignore_letters = ['?', '!', ',', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = word_tokenize(pattern)
        words.extend(word_list)      
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) # 1 if word is present in sentence of set of words else 0.
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1 # making 1 at index of class of current pattern
    training.append([bag, output_row])

random.shuffle(training)
count = 0
training = np.array(training, dtype = object) # for jagged arrays

X_train = list(training[:,0])
y_train = list(training[:,1])
# print("X_train : ", X_train, "y_train : ",  y_train)
print(np.array(X_train[0]).shape)

model = Sequential()

model.add(InputLayer(shape = (len(X_train[0]),)))
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0])))
model.add(Activation("softmax"))

sgd = SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss= 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'] )
model.fit(np.array(X_train), 
          np.array(y_train),
          epochs = 50,
          batch_size = 5,
          verbose = 1)

model.save('chatbot_model.keras')
print("Done!")

