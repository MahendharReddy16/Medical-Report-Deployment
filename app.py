import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sklearn
import tqdm
import nltk
import cv2
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image
import time

import tensorflow as tf
import keras
from keras.layers import Input,concatenate,Dropout,LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from keras import Model
from tensorflow.keras import activations
import warnings
warnings.filterwarnings("ignore")
import nltk.translate.bleu_score as bleu
from tensorflow.keras.models import load_model
from helpers import *
import cv2
import flask
from flask import Flask, request,jsonify, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from PIL import Image
import os
#loading the pretrained chexnet model for getting image features
final_chexnet_model=load_model("chexnet_final.h5")
#loading embedded matrix
embedding_matrix=np.load('Embedding_matrix.npy')
print(embedding_matrix.shape)
#Loading the pretrained tokenizer
import pickle
with open('tokenizer_1.pkl', 'rb') as handle:
    token= pickle.load(handle)
#loading the dict file with image names and corresponding reports to calculate bleu scores
with open('data_dict.pkl', 'rb') as handle:
    Data= pickle.load(handle)
app = Flask(__name__)


@app.route('/')
def main_page():
    
    return flask.render_template('login.html')



def beam_search(image_features,beam_index):

    '''this function will take the input images and beam _index from the predict function and extracts the features and gives the predicted      report for the given images'''
      
    enc_units=64
    embedding_dim=300
    dec_units=64
    att_units=64
    max_len = 100
    bs = 10
    vocab_size = 1427
    model  = encoder_decoder(enc_units,embedding_dim,vocab_size,max_len,dec_units,att_units,bs)
    #loading the weights
    model.load_weights("model_3/wts")

    hidden_state =  tf.zeros((1, enc_units))
    #image_features=image_feature_extraction(image_1,image_2)

    def take_second(elem):
        return elem[1]

    encoder_out = model.layers[0](image_features)

    start_token = [token.word_index["<sos>"]]
    dec_word = [[start_token, 0.0]]
    while len(dec_word[0][0]) < max_len:
      temp = []
      for word in dec_word:

          predict, hidden_state,alpha = model.layers[1].onestep(tf.expand_dims([word[0][-1]],1), encoder_out, hidden_state)


          word_predict = np.argsort(predict[0])[-beam_index:]
          for i in word_predict:

              next_word, probab = word[0][:], word[1]
              next_word.append(i)
              probab += predict[0][i] 
              temp.append([next_word, probab.numpy()])
      dec_word = temp
      # Sorting according to the probabilities scores


      dec_word = sorted(dec_word, key=take_second)

      # Getting the top words
      dec_word = dec_word[-beam_index:] 


    final = dec_word[-1]

    report =final[0]
    score = final[1]
    temp = []

    for word in report:
        if word!=0:
            if word != token.word_index['<eos>']:
                temp.append(token.index_word[word])
            else:
                break 

    rep = ' '.join(e for e in temp)        

    return rep, score

class encoder_decoder(tf.keras.Model):
        
    def __init__(self,enc_units,embedding_dim,vocab_size,output_length,dec_units,att_units,batch_size):
        super().__init__()


        self.batch_size=batch_size
        self.encoder =Encoder(enc_units)
        self.decoder=Decoder(vocab_size,embedding_dim,output_length,dec_units,att_units)

    def call(self, data):
        features,report  = data[0], data[1]

        encoder_output= self.encoder(features)
        state_h=self.encoder.initialize_states(self.batch_size)

        output= self.decoder(report, encoder_output,state_h)

        return output
    
class Encoder(tf.keras.Model):
    def __init__(self,units):
        super().__init__()
        self.units=units

        
    def build(self,input_shape):
        self.dense1=Dense(self.units,activation="relu",kernel_initializer=tf.keras.initializers.glorot_uniform(seed = 0),name="encoder_dense")
        self.maxpool=tf.keras.layers.Dropout(0.5)

    def call(self,input_):
         enc_out=self.maxpool(input_)
         enc_out=self.dense1(enc_out) 

         return enc_out

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, output_length, dec_units,att_units):
          super().__init__()
    #Intialize necessary variables and create an object from the class onestepdecoder
          self.onestep=One_Step_Decoder(vocab_size, embedding_dim, output_length, dec_units,att_units)



    def call(self, input_to_decoder,encoder_output,state_1):

       #Initialize an empty Tensor array, that will store the outputs at each and every time step
       #Create a tensor array as shown in the reference notebook

       #Iterate till the length of the decoder input
       # Call onestepdecoder for each token in decoder_input
       # Store the output in tensorarray
       # Return the tensor array

      all_outputs=tf.TensorArray(tf.float32,input_to_decoder.shape[1],name="output_array")
      for step in range(input_to_decoder.shape[1]):
          output,state_1,alpha=self.onestep(input_to_decoder[:,step:step+1],encoder_output,state_1)

          all_outputs=all_outputs.write(step,output)
      all_outputs=tf.transpose(all_outputs.stack(),[1,0,2])

      return all_outputs

    
class Attention(tf.keras.layers.Layer):
    '''this is the attention class. 
       Here the input to the decoder and the gru hidden state at the pevious time step are given, and the context vector is calculated
       This context vector is calculated uisng the attention weights. This context vector is then passed to the decoder model
       Here conact function is used for calaculating the attention weights'''

    def __init__(self,att_units):

        super().__init__()

        self.att_units=att_units

    def build(self,input_shape):
        self.wa=tf.keras.layers.Dense(self.att_units)
        self.wb=tf.keras.layers.Dense(self.att_units)
        self.v=tf.keras.layers.Dense(1)


    def call(self,decoder_hidden_state,encoder_output):

        x=tf.expand_dims(decoder_hidden_state,1)

        alpha_dash=self.v(tf.nn.tanh(self.wa(encoder_output)+self.wb(x)))

        alphas=tf.nn.softmax(alpha_dash,1)

        context_vector=tf.matmul(encoder_output,alphas,transpose_a=True)[:,:,0]
        # context_vector = alphas*encoder_output
        # print("c",context_vector.shape)


        return (context_vector,alphas)

class One_Step_Decoder(tf.keras.Model):
    '''This class will perform the decoder task.
       The main decoder will call this onestep decoder at every time step. This one step decoder in turn class the atention model and    return the ouptput at time step t.
       This output is passed through the final softmax layer with output size =vocab size, and pass this result to the main decoder model'''
    def __init__(self,vocab_size, embedding_dim, input_length, dec_units ,att_units):

        # Initialize decoder embedding layer, LSTM and any other objects needed
        super().__init__()

        self.att_units=att_units
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.input_length=input_length

        self.dec_units=dec_units
        self.attention=Attention(self.att_units)
        #def build(self,inp_shape):
        self.embedding=tf.keras.layers.Embedding(self.vocab_size,output_dim=self.embedding_dim,
                                         input_length=self.input_length,mask_zero=True,trainable=False,weights=[embedding_matrix])

        self.gru=tf.keras.layers.GRU(self.dec_units,return_sequences=True,return_state=True,
                                                recurrent_initializer='glorot_uniform',name="decoder_gru")
        self.dense=tf.keras.layers.Dense(self.vocab_size,name="decoder_final_dense") 
        self.dense_2=tf.keras.layers.Dense(self.embedding_dim,name="decoder_dense2") 


    def call(self,input_to_decoder, encoder_output, state_h):
    
            embed=self.embedding(input_to_decoder)

            context_vector,alpha=self.attention(state_h,encoder_output)

            context_vector=self.dense_2(context_vector) 

            result=tf.concat([tf.expand_dims(context_vector, axis=1),embed],axis=-1)


            output,decoder_state_1=self.gru(result,initial_state=state_h)
            out=tf.reshape(output,(-1,output.shape[-1]))

            out=tf.keras.layers.Dropout(0.5)(out)

            dense_op=self.dense(out)

            return dense_op,decoder_state_1,alpha

def image_feature_extraction(image_1,image_2):
    
    ''' takes the images as input and returns the features extracted  from the  pretrained chexnet model'''
    image_1 = Image.open(image_1)
    image_1.show()
    image_1= np.asarray(image_1.convert("RGB"))

    image_2=Image.open(image_2)
    image_2.show()
    image_2 = np.asarray(image_2.convert("RGB"))

    #normalising the image 
    image_1=image_1/255
    image_2=image_2/255

    #resize the image into (224,224)
    image_1 = cv2.resize(image_1,(224,224))
    image_2 = cv2.resize(image_2,(224,224))

    image_1= np.expand_dims(image_1, axis=0)
    image_2= np.expand_dims(image_2, axis=0)

  #now we have read two image per patient. this is goven to the chexnet model for feature extraction

    image_1_out=final_chexnet_model(image_1)
    image_2_out=final_chexnet_model(image_2)
    #conactenate along the width
    conc=np.concatenate((image_1_out,image_2_out),axis=2)
    #reshape into(no.of images passed, length*breadth, depth)
    image_feature=tf.reshape(conc, (conc.shape[0], -1, conc.shape[-1]))

    return image_feature

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    image_link = [x for x in request.form.values()]
    
    final_features = image_feature_extraction(image_link[0],image_link[1])
    prediction,score = beam_search(final_features,3)
    
    return render_template('login.html', prediction_text=prediction)
if __name__ == '__main__':
    app.run(debug=True)

    