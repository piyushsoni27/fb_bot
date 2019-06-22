# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:58:35 2019

@author: p.soni
"""

from keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model

import numpy as np

import nmt_utils as nmt

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a,s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    
    return context

def create_model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    print("Creating encoder-decoder model")
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    post_activation_LSTM_cell = LSTM(n_s, return_state = True)
    output_layer = Dense(machine_vocab_size, activation=nmt.softmax)
    
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    
    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context,initial_state=[s, c])
        
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)
        
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
    
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    print("Model created!!")
    return model

def myModel(Xoh, Yoh, m, Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size, lr = 0.005, beta_1=0.9, beta_2=0.999, decay = 0.01):   
    model = create_model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size)
    
    opt = Adam (lr, beta_1, beta_2, decay) 
    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics=["accuracy"])
    
    outputs = list(Yoh.swapaxes(0,1))
    
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    
    model_history = model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
    
    print("Saving weights...")
    #model.save_weights( 'model_weights_1_epoch.h5')
    print("Weight Saved!")
    return model, model_history

def main():

    params1 = {'m' : 1000, 
              'n_a' : 32,
              'n_s' : 64,
              'Tx' : 30,   # Max length of input
              'Ty' : 10    # o/p length "YYYY-MM-DD"
              }    
    
    dataset, human_vocab, machine_vocab, inv_machine_vocab = nmt.load_dataset(params1['m'])
    X, Y, Xoh, Yoh = nmt.preprocess_data(dataset, human_vocab, machine_vocab, params1['Tx'], params1['Ty'])
    
    params2 = {'machine_vocab_size' : len(machine_vocab),
               'human_vocab_size' : len(human_vocab)}
    
    params = {**params1, **params2}
    
    hparams = {'lr' : 0.005, 
               'beta_1' : 0.9, 
               'beta_2' : 0.999, 
               'decay' : 0.01}
    
    vocab = {'human_vocab' : human_vocab,
             'machine_vocab' : machine_vocab,
             'inv_machine_vocab' : inv_machine_vocab}
    
    # Defined shared layers as global variables
    global repeator, concatenator, densor1, densor2, activator, dotor, encoder
    
    encoder = Bidirectional(LSTM(params['n_a'], return_sequences=True))
    
    repeator = RepeatVector(params['Tx'])
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(nmt.softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded from nmt_utils
    dotor = Dot(axes = 1)
    
    model, _ = myModel(Xoh, Yoh, **params, **hparams)
    model.load_weights('models/coursera_model.h5')
    
    return model, params, vocab

if __name__ == '__main__':
    
    model, params, vocab = main()
    
    s0 = np.zeros((params['m'], params['n_s']))
    c0 = np.zeros((params['m'], params['n_s']))

    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
    for example in EXAMPLES:
        
        source = nmt.string_to_int(example, params['Tx'], vocab['human_vocab'])
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=params['human_vocab_size']), source)))
        source = source.reshape((1, source.shape[0], source.shape[1]))
        prediction = model.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis = -1)
        output = [vocab['inv_machine_vocab'][int(i)] for i in prediction]
        
        print("source:", example)
        print("output:", ''.join(output))
