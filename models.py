#!/usr/bin/python3

import numpy as np
import tensorflow as tf

def RWKVAttention(hidden_size = 768, head_size = 64):
  hidden = tf.keras.Input((None, hidden_size)) # hidden.shape = (batch, seq_len, hidden)
  attn_x = tf.keras.Input((hidden_size,))
  attn_kv = tf.keras.Input((hidden_size // head_size, head_size, head_size,))
  ffn_x = tf.keras.Input((hidden_size,))
  if 
  # extract key value
  shifted = 

def RWKVBlock(hidden_size = 768, head_size = 64, seq_mode = True):
  hidden = tf.keras.Input((None, hidden_size)) # hidden.shape = (batch, seq_len, hidden)
  attn_x = tf.keras.Input((hidden_size,))
  attn_kv = tf.keras.Input((hidden_size // head_size, head_size, head_size,))
  ffn_x = tf.keras.Input((hidden_size,))
  hidden = tf.keras.layers.LayerNormalization()(hidden)
  

def RWKV(vocab_size, hidden_size = 768, use_cache = True, num_hidden_layers = 12, head_size = 64, seq_mode = True):
  inputs = tf.keras.Input((None,), dtype = tf.int32)
  if use_cache:
    state_attn_x = tf.keras.Input((hidden_size, num_hidden_layers))
    state_attn_kv = tf.keras.Input((hidden_size // head_size, head_size, head_size, num_hidden_layers))
    state_ffn_x = tf.keras.Input((hidden_size, num_hidden_layers))
  else:
    state_attn_x = tf.keras.layers.Lambda(lambda x, h, l: tf.zeros((tf.shape(x)[0], h, l), dtype = tf.float32), arguments = {'h': hidden_size, 'l': num_hidden_layers})(inputs)
    state_attn_kv = tf.keras.layers.Lambda(lambda x, hn, hs, l: tf.zeros((tf.shape(x)[0], hn, hs, hs, l), dtype = tf.float32), arguments = {'hn': hidden_size // head_size, 'hs': head_size, 'l': num_hidden_layers})(inputs)
    state_ffn_x = tf.keras.layers.Lambda(lambda x, h, l: tf.zeros(tf.shape(x)[0], h, l, dtype = tf.float32), arguments = {'h': hidden_size, 'l': num_hidden_layers})(inputs)
  hidden_states = tf.keras.layers.Embedding(vocab_size, hidden_size)(inputs)
  hidden_states = tf.keras.layers.LayerNormalization()(hidden_states)
  for i in range(num_hidden_layers):
    attn_x = tf.keras.layers.Lambda(lambda x, i: x[...,i], arguments = {'i': i})(state_attn_x)
    attn_kv = tf.keras.layers.Lambda(lambda x, i: x[...,i], arguments = {'i': i})(state_attn_kv)
    ffn_x = tf.keras.layers.Lambda(lambda x, i: x[...,i], arguments = {'i': i})(state_ffn_x)

