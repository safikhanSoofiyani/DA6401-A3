# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:55:30 2022

@author: safik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import tensorflow as tf
import keras as ks
import random
import csv
import argparse

from matplotlib import pyplot, font_manager
from seaborn import heatmap 
from tqdm import tqdm

from keras.layers import Dense, Input,InputLayer,Flatten,Activation,LSTM,SimpleRNN,GRU,TimeDistributed,Embedding
from keras.models import Sequential, Model,load_model
from keras.callbacks import EarlyStopping
from keras.layers import Concatenate, AdditiveAttention

from sys import argv

random.seed(hash("seriously you compete with me") % 2**32 - 1)
np.random.seed(hash("i am mohammed safi") % 2**32 - 1)
tf.random.set_seed(hash("ur rahman khan") % 2**32 - 1)

import wandb

from wandb.keras import WandbCallback



def load_data():
    
    train_file_path = "te.translit.sampled.train.tsv"
    val_file_path = "te.translit.sampled.dev.tsv"



    train_tsv_file = open(train_file_path, "r", encoding="utf-8")
    train_dataset = csv.reader(train_tsv_file, delimiter="\t")

    val_tsv_file = open(val_file_path, "r", encoding="utf-8")
    val_dataset = csv.reader(val_tsv_file, delimiter="\t")

    return train_dataset, val_dataset


def load_test_data():

    test_file_path = "te.translit.sampled.test.tsv"

    test_tsv_file = open(test_file_path, "r", encoding="utf-8")
    test_dataset = csv.reader(test_tsv_file, delimiter="\t")

    return test_dataset

def prepare_data():

    train_dataset, val_dataset = load_data()
    
    input = []
    target = []
    #print(english)
    for i in train_dataset:
        #print(i) 
        target.append(i[0])
        #print(i[1])
        input.append(i[1])
    #print(english)
    target = np.array(target)
    input = np.array(input)

    # Validation data
    val_input = []
    val_target = []

    for i in val_dataset:
        val_target.append(i[0])
        val_input.append(i[1])

    val_target = np.array(val_target)
    val_input = np.array(val_input)

    for i in range(len(target)):
        target[i] = "\t" + target[i] + "\n"
    
    for i in range(len(val_target)):
        val_target[i] = "\t" + val_target[i] + "\n"

    return input, target, val_input, val_target


def prepare_test_data():

    test_dataset = load_test_data()
    
    test_input = []
    test_target = []
    #print(english)
    for i in test_dataset:
        #print(i) 
        test_target.append(i[0])
        #print(i[1])
        test_input.append(i[1])
    #print(english)
    target = np.array(test_target)
    input = np.array(test_input)

    for i in range(len(target)):
        test_target[i] = "\t" + target[i] + "\n"
    


    return test_input, test_target


def one_hot_encoding_test(input, target, input_tokens, target_tokens):

    input_token_index = dict([(char, i) for i, char in enumerate(input_tokens)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_tokens)])

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    
    encoder_input_data = np.zeros(
        (len(input), max_encoder_seq_length), dtype = "float32" )
    decoder_input_data = np.zeros(
        (len(input), max_decoder_seq_length), dtype = "float32" )
    decoder_target_data = np.zeros(
        (len(input), max_decoder_seq_length, num_decoder_tokens), dtype = "float32" )
    
    for i, (inp, tar) in enumerate(zip(input, target)):
        for t, char in enumerate(inp):
            encoder_input_data[i,t] = input_token_index[char]

        for t, char in enumerate(tar):
            decoder_input_data[i,t] = target_token_index[char]
            if t>0:
                decoder_target_data[i,t-1,target_token_index[char]] = 1.0


    return encoder_input_data, decoder_input_data, decoder_target_data



def getTokens(input, target, val_input, val_target):
    # Getting input and target language characters

    # Training set
    input_tokens = set()
    target_tokens = set()

    for word in input:
        for char in word:
            if char not in input_tokens:
                input_tokens.add(char)

    for word in target:
        for char in word:
            if char not in target_tokens:
                target_tokens.add(char)

    # Validation set
    val_input_tokens = set()
    val_target_tokens = set()

    for word in val_input:
        for char in word:
            if char not in val_input_tokens:
                val_input_tokens.add(char)

    for word in val_target:
        for char in word:
            if char not in val_target_tokens:
                val_target_tokens.add(char)

    input_tokens = sorted(list(input_tokens))
    target_tokens = sorted(list(target_tokens))
    
    return input_tokens, target_tokens, val_input_tokens, val_target_tokens



def one_hot_encoding(input, target, val_input, val_target, input_tokens, target_tokens):

    input_token_index = dict([(char, i) for i, char in enumerate(input_tokens)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_tokens)])

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    
    max_encoder_seq_length = max([len(txt) for txt in input])
    max_decoder_seq_length = max([len(txt) for txt in target])
    
    num_encoder_tokens = len(input_tokens)
    num_decoder_tokens = len(target_tokens)

    
    encoder_input_data = np.zeros(
        (len(input), max_encoder_seq_length), dtype = "float32" )
    decoder_input_data = np.zeros(
        (len(input), max_decoder_seq_length), dtype = "float32" )
    decoder_target_data = np.zeros(
        (len(input), max_decoder_seq_length, num_decoder_tokens), dtype = "float32" )
    
    for i, (inp, tar) in enumerate(zip(input, target)):
        for t, char in enumerate(inp):
            encoder_input_data[i,t] = input_token_index[char]

        for t, char in enumerate(tar):
            decoder_input_data[i,t] = target_token_index[char]
            if t>0:
                decoder_target_data[i,t-1,target_token_index[char]] = 1.0

    

    encoder_val_input_data = np.zeros(
        (len(val_input), max_encoder_seq_length), dtype = "float32" )
    decoder_val_input_data = np.zeros(
        (len(val_input), max_decoder_seq_length), dtype = "float32" )
    decoder_val_target_data = np.zeros(
        (len(val_input), max_decoder_seq_length, num_decoder_tokens), dtype = "float32" )

    for i, (inp, tar) in enumerate(zip(val_input, val_target)):
        for t, char in enumerate(inp):
            encoder_val_input_data[i,t] = input_token_index[char]

        for t, char in enumerate(tar):
            decoder_val_input_data[i,t] = target_token_index[char]
            if t>0:
                decoder_val_target_data[i,t-1,target_token_index[char]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data, encoder_val_input_data, decoder_val_input_data, decoder_val_target_data




def rnn( num_encoders, embed_size, dropout, num_decoders, hidden_layer_size):
  # e_in : Encoder input
  # e_out : Encoder output
  # e_states: Encoder states
  # d_in : Decoder input
  # d_out : Decoder output
  # d_dense : Dense layer for decoder

  enc_in = Input(shape=(max_encoder_seq_length,), name="encoder_input")
  enc_out = Embedding(num_encoder_tokens, embed_size, trainable=True, name = "encoder_embedding")(enc_in)
  #enc_out = enc_in

  enc_layers = []
  enc_states = []

  for i in range(num_encoders):
        encoder = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="encoder_rnn"+str(i+1))
        enc_layers.append(encoder)
        enc_out, state_h = encoder(enc_out)
        enc_states.append([state_h])

 
  
  dec_in = Input(shape=(max_decoder_seq_length,), name="decoder_input")
  dec_out = Embedding(num_decoder_tokens, embed_size, trainable=True, name="decoder_embedding")(dec_in)

  dec_layers = []

  for i in range(num_decoders):
        decoder = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="decoder_rnn"+str(i+1))
        dec_layers.append(decoder)
        dec_out, _ = decoder(dec_out, initial_state = enc_states[i])

  
  
  # Adding Attention:
  decoder_attn=AdditiveAttention(name="attention_layer")
  decoder_concat=Concatenate(name="concatenate_layer")
  cont_vec,attn_wts=decoder_attn([dec_out,enc_out],return_attention_scores=True)
  dec_out= decoder_concat([dec_out,cont_vec])
  
  dec_dense =TimeDistributed(Dense(num_decoder_tokens, activation="softmax", name="dense_softmax"))
  dec_out = dec_dense(dec_out)
  
  model = Model([enc_in, dec_in], dec_out)

  return model, enc_layers, dec_layers


def lstm( num_encoders, embed_size, dropout, num_decoders, hidden_layer_size):
  # e_in : Encoder input
  # e_out : Encoder output
  # e_states: Encoder states
  # d_in : Decoder input
  # d_out : Decoder output
  # d_dense : Dense layer for decoder
  
  enc_in = Input(shape=(max_encoder_seq_length,))
  enc_out = Embedding(num_encoder_tokens, embed_size, trainable=True)(enc_in)

  enc_layers = []
  enc_states = []

  for i in range(num_encoders):
        encoder = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout)
        enc_layers.append(encoder)
        enc_out, state_h, state_c = encoder(enc_out)
        enc_states.append([state_h, state_c])

  
  dec_in = Input(shape=(max_decoder_seq_length,))
  dec_out = Embedding(num_decoder_tokens, embed_size, trainable=True)(dec_in)

  dec_layers = []

  for i in range(num_decoders):
        decoder = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout)
        dec_layers.append(decoder)
        dec_out, _, _ = decoder(dec_out, initial_state = enc_states[i])


  
  # Adding Attention:
  decoder_attn=AdditiveAttention()
  decoder_concat=Concatenate()
  cont_vec,attn_wts=decoder_attn([dec_out,enc_out],return_attention_scores=True)
  dec_out= decoder_concat([dec_out,cont_vec])
  
  dec_dense = TimeDistributed(Dense(num_decoder_tokens, activation="softmax"))
  dec_out = dec_dense(dec_out)
  model = Model([enc_in, dec_in], dec_out)

  return model, enc_layers, dec_layers


def gru(num_encoders, embed_size, dropout, num_decoders, hidden_layer_size):
  # e_in : Encoder input
  # e_out : Encoder output
  # e_states: Encoder states
  # d_in : Decoder input
  # d_out : Decoder output
  # d_dense : Dense layer for decoder
  
 
  enc_in = Input(shape=(max_encoder_seq_length,))
  enc_out = Embedding(num_encoder_tokens, embed_size, trainable=True)(enc_in)

  enc_layers = []
  enc_states = []

  for i in range(num_encoders):
        encoder = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout)
        enc_layers.append(encoder)
        enc_out, state_h = encoder(enc_out)
        enc_states.append([state_h])
  
  
  dec_in = Input(shape=(max_decoder_seq_length,))
  dec_out = Embedding(num_decoder_tokens, embed_size, trainable=True)(dec_in)

  dec_layers = []

  for i in range(num_decoders):
        decoder = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout)
        dec_layers.append(decoder)
        dec_out, _ = decoder(dec_out, initial_state = enc_states[i])
  
    
  # Adding Attention:
  decoder_attn=AdditiveAttention()
  decoder_concat=Concatenate()
  cont_vec,attn_wts=decoder_attn([dec_out,enc_out],return_attention_scores=True)
  dec_out= decoder_concat([dec_out,cont_vec])
  
  
  dec_dense = TimeDistributed(Dense(num_decoder_tokens, activation="softmax"))
  dec_out = dec_dense(dec_out)
  model = Model([enc_in, dec_in], dec_out)

  return model, enc_layers, dec_layers



def build_model(num_encoders, num_decoders, cell, embed_size, dropout, hidden_layer_size):


  if cell == "RNN":
    model, enc_layers, dec_layers=rnn( num_encoders, embed_size, dropout, num_decoders, hidden_layer_size)
    return model, enc_layers, dec_layers
  elif cell == "LSTM":
    model, enc_layers, dec_layers=lstm(num_encoders, embed_size, dropout, num_decoders, hidden_layer_size)
    return model, enc_layers, dec_layers
  elif cell == "GRU":
    model, enc_layers, dec_layers=gru(num_encoders, embed_size, dropout, num_decoders, hidden_layer_size)
    return model, enc_layers, dec_layers


def inferencing(model, num_encoders, num_decoders, enc_layers, dec_layers, cell, hidden_layer_size):


    # ENCODER MODEL RECONSTRUCTION 
    enc_in = model.input[0]  # input_1
    EmbeddingLayer = model.layers[2]    
    enc_out = EmbeddingLayer(enc_in)

    enc_states = []

    if cell == 'RNN':
        for i in range(num_encoders):
            enc_out, state_h = enc_layers[i](enc_out)
            enc_states += [state_h] 
    elif cell == 'LSTM':
        for i in range(num_encoders):
            enc_out, state_h, state_c = enc_layers[i](enc_out)
            enc_states += [state_h, state_c]   
    elif cell == 'GRU':
        for i in range(num_encoders):
            enc_out, state_h = enc_layers[i](enc_out)
            enc_states += [state_h] 

    encoder = Model(enc_in, enc_states + [enc_out])


    

    dec_in = model.input[1]    
    EmbeddingLayer = model.layers[3] 
    dec_out = EmbeddingLayer(dec_in)

    dec_states = []
    dec_initial_states = []
    
    if cell == 'RNN' :
        for i in range(num_decoders):
            dec_initial_states += [Input(shape=(hidden_layer_size,))]
            dec_out, state_h = dec_layers[i](dec_out, initial_state = dec_initial_states[i])
            dec_states += [state_h]
    elif cell == "LSTM":
        j=0
        for i in range(num_decoders):
            dec_initial_states += [Input(shape=(hidden_layer_size, )) , Input(shape=(hidden_layer_size, ))]
            dec_out, state_h, state_c = dec_layers[i](dec_out, initial_state=dec_initial_states[i+j:i+j+2])
            dec_states += [state_h , state_c]
            j += 1
    elif cell == "GRU":
        for i in range(num_decoders):
            dec_initial_states += [Input(shape=(hidden_layer_size,))]
            dec_out, state_h = dec_layers[i](dec_out, initial_state = dec_initial_states[i])
            dec_states += [state_h]

    total_encoder_decoders = (2 * num_encoders)
    attention_layer = model.layers[4 + total_encoder_decoders]
    attention_input = Input(shape=(max_encoder_seq_length,hidden_layer_size))   

    context_vec, attention_weights = attention_layer([dec_out, attention_input], return_attention_scores=True)
    
    concat_layer = model.layers[5 + total_encoder_decoders]
    dec_out = concat_layer([dec_out, context_vec])

    dec_dense = model.layers[6 + total_encoder_decoders]
    dec_out = dec_dense(dec_out)
    decoder = Model([dec_in] + dec_initial_states + [attention_input], [dec_out] + dec_states + [attention_weights])

    return encoder, decoder

def decode_sequence(input_seq, encoder, decoder):
    # Encode the input as state vectors.
    states_value = encoder.predict(input_seq)
    attention_input = states_value[-1]

    states_value = states_value[:-1]
    
    target_seq = np.zeros((1, 1)) 
    target_seq[0, 0] = target_token_index["\t"]
    
    attention_weights = []
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens = decoder.predict([target_seq] + states_value + [attention_input])
        sampled_token_index = np.argmax(output_tokens[0][0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = output_tokens[1:-1]
        attention_weights.append(output_tokens[-1][0][0])
        
    return decoded_sentence, attention_weights


def train():
  default_config={
      "cell": "RNN",
      "embed_size":256,
      "dropout":0.2,
      "num_layers": 1,
      "hidden_layer_size":128,
      "epochs": 1,
      "batch_size": 64,
      "optimizer": 'adam'
  }

  wandb.init(config=default_config, project=project_name, entity=entity_name)
  config = wandb.config
  wandb.run.name = "ATT"+str(config.cell)\
      +"_"+str(config.embed_size)\
      +"_"+str(config.dropout)\
      +"_"+str(config.num_layers)\
      +"_"+str(config.num_layers)\
      +"_"+str(config.hidden_layer_size)\
      +"_"+str(config.epochs)\
      +"_"+str(config.batch_size)

  #wandb.run.save()
  
  
  batch_size = config['batch_size']
  cell = config['cell']
  embed_size = config['embed_size']
  dropout = config['dropout']
  num_encoders = config['num_layers']
  num_decoders = config['num_layers']
  hidden_layer_size = config['hidden_layer_size']
  epochs = config['epochs']
  optimizer = config['optimizer']


  model, encoder_layers, decoder_layers = build_model(num_encoders, num_decoders, cell, embed_size, dropout, hidden_layer_size)

  
  model.compile(
      optimizer = optimizer,
      loss = tf.keras.losses.CategoricalCrossentropy(name='loss'),
      metrics = [tf.keras.metrics.CategoricalAccuracy(name='acc')]
  )
  
  model.fit(
      [encoder_input_data, decoder_input_data],
      decoder_target_data,
      batch_size = batch_size,
      epochs = epochs,
      shuffle = True,
      callbacks=[WandbCallback()],
      validation_data= ([encoder_val_input_data, decoder_val_input_data], decoder_val_target_data)
  )

  wandb.run.finish()
  
  
hyperparameters = {

          "cell":{
              "values":["RNN","GRU","LSTM"]
              },

          "embed_size":{
              "values":[16,32,64,256]
              },

          "hidden_layer_size":{
              "values":[16,32,64,256]
              },

          "num_layers":{
              "values":[1,2,3]
              },

          "dropout":{
              "values":[0.2,0.3,0.4]
              },

          "epochs":{
              "values":[5,10,15,20]
              },

          "batch_size":{
              "values":[32,64]
              },

          "optimizer":{
              "values":["adam", "rmsprop", "nadam"]
              }
      }



def sweeper(project_name,entity_name):
  sweep_config={

      "method": "bayes",
      "metric": {
          "name": "val_acc", 
          "goal": "maximize"
          },
      "parameters":hyperparameters
  }

  sweep_id=wandb.sweep(sweep_config, project=project_name, entity=entity_name)
  wandb.agent(sweep_id,train)




best_hyperparameters = {
      "cell": "LSTM",
      "embed_size":256,
      "dropout":0.4,
      "num_layers": 2,
      "hidden_layer_size":256,
      "epochs": 20,
      "batch_size": 64,
      "optimizer": 'adam'
}







def testing(entity_name, project_name):

    wandb.init(config=best_hyperparameters, project=project_name, entity=entity_name)
    config = wandb.config
    wandb.run.name = "TestRun - Attention"


    batch_size = config['batch_size']
    cell = config['cell']
    embed_size = config['embed_size']
    dropout = config['dropout']
    num_encoders = config['num_layers']
    num_decoders = config['num_layers']
    hidden_layer_size = config['hidden_layer_size']
    epochs = config['epochs']
    optimizer = config['optimizer']

    model, encoder_layers, decoder_layers =  build_model(num_encoders, num_decoders, cell, embed_size, dropout, hidden_layer_size )

    model.compile(optimizer = optimizer,
      loss = tf.keras.losses.CategoricalCrossentropy(name='loss'),
      metrics = [tf.keras.metrics.CategoricalAccuracy(name='acc')]
    )

    model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    shuffle = True,
    callbacks=[WandbCallback()]
    )


    n = len(test_target)
    encoder_model, decoder_model = inferencing(model, best_hyperparameters['num_layers'], best_hyperparameters['num_layers'], encoder_layers, decoder_layers, best_hyperparameters['cell'], best_hyperparameters['hidden_layer_size'])
    rows = []
    attentions = []
    match = 0
    for i in tqdm(range(n)):
        input = encoder_test_input_data[i:i+1]
        output, attn_weights = decode_sequence(input,encoder_model, decoder_model)
        expected = test_target[i]
        attentions.append(attn_weights)

        if output.strip() == expected.strip():
            match += 1

        point = [test_input[i], output.strip()]
        rows.append(point)

    print("Test Accuracy: ",match/n)
    wandb.log({"TestAccuracy" : match/n})

    wandb.run.finish()

    return model, rows, attentions





def testing_model(batch_size, cell, embed_size, dropout, num_layers, hidden_layer_size, epochs, optimizer):

 
    num_encoders = num_layers
    num_decoders = num_layers

    model, encoder_layers, decoder_layers =  build_model(num_encoders, num_decoders, cell, embed_size, dropout, hidden_layer_size )

    model.compile(optimizer = optimizer,
      loss = tf.keras.losses.CategoricalCrossentropy(name='loss'),
      metrics = [tf.keras.metrics.CategoricalAccuracy(name='acc')]
    )

    model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    shuffle = True,
    )


    n = len(test_target)
    encoder_model, decoder_model = inferencing(model, num_layers, num_layers, encoder_layers, decoder_layers, cell, hidden_layer_size)
    rows = []
    attentions = []
    match = 0
    for i in tqdm(range(n)):
        input = encoder_test_input_data[i:i+1]
        output, attn_weights = decode_sequence(input,encoder_model, decoder_model)
        expected = test_target[i]
        attentions.append(attn_weights)

        if output.strip() == expected.strip():
            match += 1

        point = [test_input[i], output.strip()]
        rows.append(point)

    print("Test Accuracy: ",match/n)
    #wandb.log({"TestAccuracy" : match/n})

    #wandb.run.finish()
    
    fields = ['Input', 'Predicted Target']

    with open('Predicted_With_Attention.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)

    return model, rows, attentions


if __name__ == "__main__":
    
    entity_name = "safikhan"
    project_name = "assgn3 trial"
    
    #parsing the various command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, required=True, help="Do you want to sweep or not: Enter 'yes' or 'no'")
    
    parser.add_argument('--cell', type=str, required=('no' in argv), help="Cell Type: LSTM or RNN or GRU")
    parser.add_argument('--embedSize', type=int, required=('no' in argv), help="Input Embedding Size: integer value")
    parser.add_argument('--dropout', type=float, required=('no' in argv), help="Dropout: float value")
    parser.add_argument('--numLayers', type=int, required=('no' in argv), help="Number of Encoder/Decoder Layers: integer value")
    parser.add_argument('--hiddenLayerSize', type=int, required=('no' in argv), help="Hidden units in cell: integer value")
    parser.add_argument('--numEpochs', type=int, required=('no' in argv), help="Number of Epochs: integer value")
    parser.add_argument('--batchSize', type=int, required=('no' in argv), help="Batch Size: integer value")
    parser.add_argument('--optimizer', type=str, required=('no' in argv), help="Optimizer function: adam or nadam or rmsprop")
    
    
    args = parser.parse_args()
    
    input, target, val_input, val_target = prepare_data()
    input_tokens, target_tokens, val_input_tokens, val_target_tokens = getTokens(input, target, val_input, val_target)

    num_encoder_tokens = len(input_tokens)
    num_decoder_tokens = len(target_tokens)

    max_encoder_seq_length = max([len(txt) for txt in input])
    max_decoder_seq_length = max([len(txt) for txt in target])

    input_token_index = dict([(char, i) for i, char in enumerate(input_tokens)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_tokens)])
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    print("Number of data points:", len(input))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)
    
    encoder_input_data, decoder_input_data, decoder_target_data, encoder_val_input_data, decoder_val_input_data, decoder_val_target_data = one_hot_encoding(input, target, val_input, val_target, input_tokens, target_tokens)
    
    test_input, test_target = prepare_test_data()
    
    encoder_test_input_data, decoder_test_input_data, decoder_test_target_data = one_hot_encoding_test(test_input, test_target, input_tokens, target_tokens)
    
    
    if args.sweep == 'no':
        
        cell = args.cell
        embed_size = args.embedSize
        dropout = args.dropout
        num_layers = args.numLayers
        hidden_layer_size = args.hiddenLayerSize
        epochs = args.numEpochs
        batch_size = args.batchSize
        optimizer = args.optimizer
        
        model = testing_model(batch_size, cell, embed_size, dropout, num_layers, hidden_layer_size, epochs, optimizer)
        
    else:
        sweeper(entity_name, project_name)
        testing(entity_name, project_name)















