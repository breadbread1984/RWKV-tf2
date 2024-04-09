#!/usr/bin/python3

import numpy as np
import tensorflow as tf

class RWKVAttention(tf.keras.layers.Layer):
  def __init__(self, hidden_size = 768, head_size = 64):
    super(RWKVAttention, self).__init__()
    self.hidden_size = hidden_size
    self.head_size = head_size
  def build(self, input_shape):
    self.time_maa_x = self.add_weight(shape = (1,1,self.hidden_size), dtype = tf.float32, trainable = True, name = 'time_maa_x')
    self.time_maa_w = self.add_weight(shape = (1,1,self.hidden_size), dtype = tf.float32, trainable = True, name = 'time_maa_w')
    self.time_maa_k = self.add_weight(shape = (1,1,self.hidden_size), dtype = tf.float32, trainable = True, name = 'time_maa_k')
    self.time_maa_v = self.add_weight(shape = (1,1,self.hidden_size), dtype = tf.float32, trainable = True, name = 'time_maa_v')
    self.time_maa_r = self.add_weight(shape = (1,1,self.hidden_size), dtype = tf.float32, trainable = True, name = 'time_maa_r')
    self.time_maa_g = self.add_weight(shape = (1,1,self.hidden_size), dtype = tf.float32, trainable = True, name = 'time_maa_g')

    self.time_maa_w1 = self.add_weight(shape = (self.hidden_size, 32 * 5), dtype = tf.float32, trainable = True, name = 'time_maa_w1')
    self.time_maa_w2 = self.add_weight(shape = (5, 32, self.hidden_size), dtype = tf.float32, trainable = True, name = 'time_maa_w2')
    self.time_decay = self.add_weight(shape = (1,1,self.hidden_size), dtype = tf.float32, trainable = True, name = 'decay')
    self.time_decay_w1 = self.add_weight(shape = (self.hidden_size, 64), dtype = tf.float32, trainable = True, name = 'decay_w1')
    self.time_decay_w2 = self.add_weight(shape = (64, self.hidden_size), dtype = tf.float32, trainable = True, name = 'decay_w2')

    self.time_faaaa = self.add_weight(shape = (self.hidden_size // self.head_size, self.head_size))

    self.receptance_w = self.add_weight(shape = (self.hidden_size, self.hidden_size), dtype = tf.float32, trainable = True, name = 'receptance_w')
    self.key_w = self.add_weight(shape = (self.hidden_size, self.hidden_size), dtype = tf.float32, trainable = True, name = 'key_w')
    self.value_w = self.add_weight(shape = (self.hidden_size, self.hidden_size), dtype = tf.float32, trainable = True, name = 'value_w')
    self.gate_w = self.add_weight(shape = (self.hidden_size, self.hidden_size), dtype = tf.float32, trainable = True, name = 'gate_w')
    self.output_w = self.add_weight(shape = (self.hidden_size, self.hidden_size), dtype = tf.float32, trainable = True, name = 'output_w')
  def call(self, inputs):
    hidden, attn_x, attn_kv, ffn_x = inputs
    # hidden.shape = (batch, seq_len, hidden)
    # attn_x.shape = (batch, hidden)
    # attn_kv.shape = (batch, hidden // head, head, head)
    # ffn_x.shape = (batch, hidden)

    # 1) extract key value
    # shifted = concat(attn_x, right shifted hidden)
    shifted = tf.concat([tf.expand_dims(attn_x, axis = 1), hidden[:,0:-1,:]], axis = 1) # shifted.shape = (batch, seq_len, hidden)
    x = hidden # x.shape = (batch, seq_len, hidden)
    xx = shifted - x # xx.shape = (batch, seq_len, hidden)
    xxx = x + xx * self.time_maa_x # xxx.shape = (batch, seq_len, hidden)
    xxx = tf.transpose(tf.math.tanh(tf.reshape(tf.linalg.matmul(xxx, self.time_maa_w1), (-1, 5, 32))), (1,0,2)) # xxx.shape = (5, batch * seq_len, 32)
    xxx = tf.reshape(tf.linalg.matmul(xxx, self.time_maa_w2), (5, tf.shape(hidden)[0], tf.shape(hidden)[1], self.hidden_size)) # xxx.shape = (5, batch, seq_len, hidden_size)
    mw, mk, mv, mr, mg = xxx[0,...], xxx[1,...], xxx[2,...], xxx[3,...], xxx[4,...] # shape = (batch, seq_len, hidden_size)

    time_decay = x + xx + (self.time_maa_w + mw)
    key = x + xx + (self.time_maa_k + mk)
    value = x + xx + (self.time_maa_v + mv)
    receptance = x + xx + (self.time_maa_r + mr)
    gate = x + xx + (self.time_maa_g + mg)

    receptance = tf.linalg.matmul(receptance, self.receptance_w)
    key = tf.linalg.matmul(key, self.key_w)
    value = tf.linalg.matmul(value, self.value_w)
    gate = tf.nn.silu(tf.linalg.matmul(gate, self.gate_w))

    time_decay = tf.linalg.matmul(tf.math.tanh(tf.linalg.matmul(time_decay, self.time_decay_w1)), self.time_decay_w2)
    time_decay = self.time_decay + time_decay

    attn_x = hidden[:, -1]

    # 2) forward
    layer_state = attn_kv

    return 

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
  hidden = tf.random.normal(shape = (2, 1, 768), dtype = tf.float32)
  attn_x = tf.random.normal(shape = (2, 768), dtype = tf.float32)
  attn_kv = tf.random.normal(shape = (2, 768 // 64, 64, 64), dtype = tf.float32)
  ffn_x = tf.random.normal(shape = (2, 768,), dtype = tf.float32)
  outputs = attention([hidden, attn_x, attn_kv, ffn_x])
  print(outputs.shape)

