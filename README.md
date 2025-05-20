# Indic transliteration system using RNNs
<!-- 3 submission for the course DA6401 Fundamentals of Deep Learning. -->
<br>


Created by : **Mohammed Safi Ur Rahman Khan**

---
## Overview :
The purpose of this project were :
1. Building a transliteration system using Recurrent Neural Networks.
2. Comparing different cells such as vanilla RNN, LSTM and GRU. 
3. Implementing attention mechanism and understand how these overcome the limitations of vanilla seq2seq models.

---
## General Instructions:
1. Install the required libraries using the following command :

```python 
pip install -r requirements.txt
```
2. The project is in the given python code filed (.py) files. These contain the code to direclty train and test the model in a non interactive way.
3. For training and testing the model from command line, run the (.py) file by following the instructions given in below sections.
4. The dataset for the RNN part of the project can be found at : [RNN Dataset Link](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar)
5. The dataset for the Transformers part of the project can be found at : [Transformer Dataset Link](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres)
---
# Building and training a Seq2Seq model :
## Running the code
### Running the python file from the terminal
The python file can be run from the terminal by passing the various command line arguments. Please make sure that the three files of the dataset are as it is present in the same directory as this python file<br>
There are two modes of running this file <br> <br>
**1. Running the hyperparameter sweeps using wandb**<br>
```sh
python rnn_without_attention.py --sweep yes
```
The code will now run in the sweep mode and will enable wandb integration and log all the data in wand. Make sure you have wandb installed if you want to run in this mode. Also, change the entity name and project name in the code before running in this mode<br> <br>
**2. Running in normal mode**<br>
```sh
python rnn_without_attention.py --sweep XXX --cell XXX --embedSize XXX --dropout XXX --numLayers XXX --hiddenLayerSize XXX --numEpochs XXX --batchSize XXX --optimizer XXX
```
Replace `XXX` in above with the appropriate parameter you want to train the model with<br>
For example: 

```sh
python rnn_without_attention.py --sweep no --cell LSTM --embedSize 64 --dropout 0.2 --numLayers 1 --hiddenLayerSize 32 --numEpochs 2 --batchSize 32 --optimizer adam
```
**Description of various command line arguments**<br>
1. `--sweep` : Do you want to sweep or not: Enter 'yes' or 'no'. If this is 'yes' then below arguments are not required. Enter below arguments only if this is 'no'<br>
2. `--cell` : Cell Type: LSTM or RNN or GRU <br>
3. `--embedSize` : Input Embedding Size: integer value <br>
4. `--dropout` : Dropout: float value
5. `--numLayers` : Number of Encoder/Decoder Layers: integer value
6. `--hiddenLayerSize` : Hidden units in cell: integer value
7. `--numEpochs` : Number of Epochs: integer value
8. `--batchSize` : Batch Size: integer value
9. `--optimizer` : Optimizer function: adam or nadam or rmsprop

### Running the jupyter notebook

This can be run in a sequential manner. i.e., one cell at a time. This notebook also has the code for plotting the various images required for the project.

## General Functions Description

### 1. Getting training data :
<br/>The training and validation data in encoded form is obtained using the following code snippet:
```python
encoder_input_data, decoder_input_data, decoder_target_data, encoder_val_input_data, decoder_val_input_data, decoder_val_target_data = one_hot_encoding(input, target, val_input, val_target, input_tokens, target_tokens)
```
where :
1. `input` : array of words of the input language (i.e. English) present in training dataset
2. `target` : array of words of the target language (i.e. Telugu) present in training dataset
3. `val_input` : array of words of the input language (i.e. English) present in validation dataset
4. `val_target` : array of words of the target language (i.e. Telugu) present in validation dataset
5. `input_tokens` : list of charaters in the input language
6. `target_tokens` : list of characters in the target language

### 2. Building Model :
<br/>The following function is used to build the RNN model:
```python
build_model(num_encoders, num_decoders, cell, embed_size, dropout, hidden_layer_size)
```
where :
1.  `num_encoders` : number of encoder layers
2.  `num_decoders` : number of decoder layers
3.  `cell` : type of the rnn cell ( simpleRNN, LSTM ,GRU )
4.  `embed_size` : size of the embedding 
5.  `dropout` : % of dropout (in decimals)
6.  `hidden_layer_size` : dimensionalilty of output space

### 3. Hyperparameter sweeps :
<br/>Use the  following function to perform hyperparameter sweeps in wandb:
```python 
sweeper(project_name,entity_name)
```
where:
1. `project_name` : Enter the wandb project name
2. `entity_name` : Enter the wandb entity name

The various hyperparameters used are : 
```python 
hyperparameters = {
          "cell":   {"values":  ["RNN","GRU","LSTM"]},
          "embed_size": {"values":   [16,32,64,256]},
          "hidden_layer_size":  {"values":  [16,32,64,256]},
          "num_layers": {"values":   [1,2,3]},
          "dropout":    {"values":  [0.2,0.3,0.4]},
          "epochs": {"values":   [5,10,15,20]},
          "batch_size": {"values":  [32,64]},
          "optimizer":  {"values":    ["adam", "rmsprop", "nadam"]}
      }
```

### 4. Getting Test data :
The test data in encoded form is obtained by using the following code snipppet :
```python 
encoder_test_input_data, decoder_test_input_data, decoder_test_target_data = one_hot_encoding_test(test_input, test_target, input_tokens, target_tokens)
```
where :
1. `test_input` : array of words of the input language (i.e. English) present in test dataset
2. `test_target` : array of words of the target language (i.e. Telugu) present in test dataset
3. `input_tokens` : list of charaters in the input language
4. `target_tokens` : list of characters in the target language


### 5. Testing the best performing model :
The best performing model from the hyperparameter sweeps is fed with test data. Use the following code snippet to find the test accuracy :
```python
testing()
```
---
# Building and training a Seq2Seq model with attention mechanism:
## Running the code
### Running the python file from the terminal
The python file can be run from the terminal by passing the various command line arguments. Please make sure that the three files of the dataset are as it is present in the same directory as this python file<br>
There are two modes of running this file <br> <br>
**1. Running the hyperparameter sweeps using wandb**<br>
```sh
python rnn_with_attention.py --sweep yes
```
The code will now run in the sweep mode and will enable wandb integration and log all the data in wand. Make sure you have wandb installed if you want to run in this mode. Also, change the entity name and project name in the code before running in this mode<br> <br>
**2. Running in normal mode**<br>
```sh
python rnn_with_attention.py --sweep XXX --cell XXX --embedSize XXX --dropout XXX --numLayers XXX --hiddenLayerSize XXX --numEpochs XXX --batchSize XXX --optimizer XXX
```
Replace `XXX` in above with the appropriate parameter you want to train the model with<br>
For example: 

```sh
python rnn_with_attention.py --sweep no --cell LSTM --embedSize 64 --dropout 0.2 --numLayers 1 --hiddenLayerSize 32 --numEpochs 2 --batchSize 32 --optimizer adam
```
**Description of various command line arguments**<br>
1. `--sweep` : Do you want to sweep or not: Enter 'yes' or 'no'. If this is 'yes' then below arguments are not required. Enter below arguments only if this is 'no'<br>
2. `--cell` : Cell Type: LSTM or RNN or GRU <br>
3. `--embedSize` : Input Embedding Size: integer value <br>
4. `--dropout` : Dropout: float value
5. `--numLayers` : Number of Encoder/Decoder Layers: integer value
6. `--hiddenLayerSize` : Hidden units in cell: integer value
7. `--numEpochs` : Number of Epochs: integer value
8. `--batchSize` : Batch Size: integer value
9. `--optimizer` : Optimizer function: adam or nadam or rmsprop

### Running the jupyter notebook

This can be run in a sequential manner. i.e., one cell at a time. This notebook also has the code for plotting the various images required for the project.

## General Functions Description
### 1. Getting training data :
<br/> The training and validation data in encoded form is obtained using the following code snippet:
```python
encoder_input_data, decoder_input_data, decoder_target_data, encoder_val_input_data, decoder_val_input_data, decoder_val_target_data = one_hot_encoding(input, target, val_input, val_target, input_tokens, target_tokens)
```
where :
1. `input` : array of words of the input language (i.e. English) present in training dataset
2. `target` : array of words of the target language (i.e. Telugu) present in training dataset
3. `val_input` : array of words of the input language (i.e. English) present in validation dataset
4. `val_target` : array of words of the target language (i.e. Telugu) present in validation dataset
5. `input_tokens` : list of charaters in the input language
6. `target_tokens` : list of characters in the target language

### 2. Building Model :
<br/> The following function is used to build the RNN model:
```python
build_model(num_encoders, num_decoders, cell, embed_size, dropout, hidden_layer_size)
```
where :
1.  `num_encoders` : number of encoder layers
2.  `num_decoders` : number of decoder layers
3.  `cell` : type of the rnn cell ( simpleRNN, LSTM ,GRU )
4.  `embed_size` : size of the embedding 
5.  `dropout` : % of dropout (in decimals)
6.  `hidden_layer_size` : dimensionalilty of output space

### 3. Hyperparameter sweeps :
<br/> Use the  following function to perform hyperparameter sweeps in wandb:
```python 
sweeper(project_name,entity_name)
```
where:
1. `project_name` : Enter the wandb project name
2. `entity_name` : Enter the wandb entity name

The various hyperparameters used are : 
```python 
hyperparameters = {
          "cell":{ "values":["RNN","GRU","LSTM"]},
          "embed_size":{ "values":[16,32,64,256]},
          "hidden_layer_size":{"values":[16,32,64,256]},
          "num_layers":{"values":[1,2,3]},
          "dropout":{"values":[0.2,0.3,0.4]},
          "epochs":{"values":[5,10,15,20]},
          "batch_size":{"values":[32,64]},
          "optimizer":{"values":["adam", "rmsprop", "nadam"]}
      }
```

### 4. Getting Test data :
<br/> The test data in encoded form is obtained by using the following code snipppet :
```python 
encoder_test_input_data, decoder_test_input_data, decoder_test_target_data = one_hot_encoding_test(test_input, test_target, input_tokens, target_tokens)
```
where :
1. `test_input` : array of words of the input language (i.e. English) present in test dataset
2. `test_target` : array of words of the target language (i.e. Telugu) present in test dataset
3. `input_tokens` : list of charaters in the input language
4. `target_tokens` : list of characters in the target language


### 5. Testing the best performing model :
<br/> The best performing model from the hyperparameter sweeps is fed with test data. Use the following code snippet to find the test accuracy :
```python
testing()
```
