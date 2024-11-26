# Bibliotecas de pré-processamento de dados de texto
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# palavras a serem ignoradas/omitidas ao enquadrar o conjunto de dados
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Biblioteca load_model
import tensorflow
from data_preprocessing import get_stem_words

# carregue o modelo
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Carregue os arquivos de dados
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):
    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words)
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []
    
    for word in words:                   # tokenize a entrada do usuário
        if word in input_word_token_2:   # converta a entrada do usuário em sua palavra-raiz: stemização
            bag_of_words.append(1)       # Remova duplicidades e classifique a entrada do usuário
        else:                            # Codificação de dados de entrada: crie a BOW para  user_input
            bag_of_words.append(0)
    bag.append(bag_of_words)
    
    return np.array(bag)
    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
 
   # extraia a classe de predicted_class_label
   predicted_class = ""

   # agora que temos a tag prevista, selecione uma resposta aleatória
   predicted_class = classes[predicted_class_label]
   for intent in intents['intents']:
    if intent['tag']==predicted_class:
       
       # selecione uma resposta aleatória do robô
        bot_response = random.choice(intent["responses"])
    
        return bot_response
    

print("Oi, eu sou a Estela, como posso ajudar?")

while True:

    # obtenha a entrada do usuário
    user_input = input('Digite sua mensagem aqui: ')

    response = bot_response(user_input)
    print("Resposta do Robô: ", response)
