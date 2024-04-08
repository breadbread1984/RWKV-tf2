#!/usr/bin/python3

import numpy as np
import tensorflow as tf

def RWKVAttention(hidden_size = 768, head_size = 64):
  hidden = tf.keras.Input((None, hidden_size)) # hidden.shape = (batch, seq_len, hidden)
  attn_x = tf.keras.Input((hidden_size,)) # attn_x.shape = (batch, hidden)
  attn_kv = tf.keras.Input((hidden_size // head_size, head_size, head_size,)) # attn_kv.shape = (batch, hidden // head, head, head)
  ffn_x = tf.keras.Input((hidden_size,))
  # extract key value
  # shifted = concat(attn_x, right shifted hidden)
  shifted = tf.keras.layers.Lambda(lambda x: tf.cond(
    tf.math.equal(tf.shape(x[1])[1], 1),
    lambda: tf.expand_dims(x[0], axis = 1),
    lambda: tf.concat([tf.expand_dims(x[0], axis = 1), x[1][:,0:-1,:]], axis = 1)
  ), output_shape = (None, None, hidden_size))([attn_x, hidden]) # shifted.shape = (batch, seq_len, hidden)
  #x = tf.keras.layers.Identity()(hidden)
  #xx = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([shifted - x]) # xx.shape = (batch, seq_len, hidden)

  return tf.keras.Model(inputs = (hidden, attn_x, attn_kv, ffn_x), outputs = shifted)

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

if __name__ == "__main__":
  attention = RWKVAttention()
  hidden = tf.random.normal(shape = (2, 2, 768), dtype = tf.float32)
  attn_x = tf.random.normal(shape = (2, 768), dtype = tf.float32)
  attn_kv = tf.random.normal(shape = (2, 768 // 64, 64, 64), dtype = tf.float32)
  ffn_x = tf.random.normal(shape = (2, 768,), dtype = tf.float32)
  outputs = attention([hidden, attn_x, attn_kv, ffn_x])
  print(outputs.shape)

