#!/usr/bin/python3

import numpy as np
import tensorflow as tf

class RWKVAttention(tf.keras.layers.Layer):
  def __init__(self, hidden_size = 768, head_size = 64):
    super(RWKVAttention, self).__init__()
    self.hidden_size = hidden_size
    self.head_size = head_size
    self.group_norm = tf.keras.layers.GroupNormalization(groups = self.hidden_size // self.head_size, epsilon = 1e-5 * (self.head_size ** 2))
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
    super(RWKVAttention, self).build(input_shape)
  def compute_output_shape(self, input_shape):
    return input_shape
  #@tf.function
  def call(self, inputs):
    hidden, attn_x, attn_kv, ffn_x = inputs
    # hidden.shape = (batch, seq_len, hidden)
    # attn_x.shape = (batch, hidden)
    # attn_kv.shape = (batch, hidden // head, head, head)
    # ffn_x.shape = (batch, hidden)

    # 1) linear to get key value
    # NOTE: time mix first time to get mix paramter for the second time
    # NOTE: time_mix = mu * shifted_x + (1 - mu) * x = mu * x_t + (1 - mu) * x_{t-1}
    x = hidden # x.shape = (batch, seq_len, hidden)
    shifted_x = tf.concat([tf.expand_dims(attn_x, axis = 1), x[:,0:-1,:]], axis = 1) # shifted_x.shape = (batch, seq_len, hidden)
    time_mix = x + (shifted_x - x) * self.time_maa_x # xxx.shape = (batch, seq_len, hidden)
    # NOTE: W2 @ tanh (W1 @ (mu * x_t + (1 - mu) * x_{t-1})), equations (11)-(15)
    xxx = tf.transpose(tf.reshape(tf.math.tanh(tf.linalg.matmul(time_mix, self.time_maa_w1)), (-1, 5, 32)), (1,0,2)) # xxx.shape = (5, batch * seq_len, 32)
    xxx = tf.reshape(tf.linalg.matmul(xxx, self.time_maa_w2), (5, tf.shape(hidden)[0], tf.shape(hidden)[1], self.hidden_size)) # xxx.shape = (5, batch, seq_len, hidden_size)
    mw, mk, mv, mr, mg = xxx[0,...], xxx[1,...], xxx[2,...], xxx[3,...], xxx[4,...] # shape = (batch, seq_len, hidden_size)
    # NOTE: time mix second time with the parameter calculated from first time
    # NOTE: value = (w + m) * shifted_x + (1 - (w + m)) * x
    time_decay = x + (shifted_x - x) * (self.time_maa_w + mw) # time_decay.shape = (batch, seq_len, hidden)
    key = x + (shifted_x - x) * (self.time_maa_k + mk) # key.shape = (batch, seq_len, hidden)
    value = x + (shifted_x - x) * (self.time_maa_v + mv) # value.shape = (batch, seq_len, hidden)
    receptance = x + (shifted_x - x) * (self.time_maa_r + mr) # receptance.shape = (batch, seq_len, hidden)
    gate = x + (shifted_x - x) * (self.time_maa_g + mg) # gate.shape = (batch, seq_len, hidden)
    # NOTE: W @ ((w + m) * shifted_x + (1 - (w + m)) * x)
    receptance = tf.linalg.matmul(receptance, self.receptance_w) # receptance.shape = (batch, seq_len, hidden)
    key = tf.linalg.matmul(key, self.key_w) # key.shape = (batch, seq_len, hidden)
    value = tf.linalg.matmul(value, self.value_w) # value.shape = (batch, seq_len, hidden)
    gate = tf.nn.silu(tf.linalg.matmul(gate, self.gate_w)) # gate.shape = (batch, seq_len, hidden)

    time_decay = tf.linalg.matmul(tf.math.tanh(tf.linalg.matmul(time_decay, self.time_decay_w1)), self.time_decay_w2) # time_decay.shape = (batch, seq_len, hidden)
    time_decay = self.time_decay + time_decay # time_decay.shape = (batch, seq_len, hidden)

    attn_x = hidden[:, -1]

    # 2) rwkv6_linear_attention
    layer_state = attn_kv # layer_state.shape = (batch, head num, head size, head size)
    time_first = self.time_faaaa # time_first.shape = (head_num, head_size)
    key = tf.transpose(tf.reshape(key, (tf.shape(key)[0], tf.shape(key)[1], self.hidden_size // self.head_size, self.head_size)), (0,2,3,1)) # key.shape = (batch, head_num, head_size, seq_len)
    value = tf.transpose(tf.reshape(value, (tf.shape(value)[0], tf.shape(value)[1], self.hidden_size // self.head_size, self.head_size)), (0,2,1,3)) # value.shape = (batch, head_num, seq_len, head_size)
    receptance = tf.transpose(tf.reshape(receptance, (tf.shape(receptance)[0], tf.shape(receptance)[1], self.hidden_size // self.head_size, self.head_size)), (0,2,1,3)) # receptance.shape = (batch, head_num, seq_len, head_size)
    time_decay = tf.transpose(tf.reshape(tf.math.exp(-tf.math.exp(time_decay)), (tf.shape(time_decay)[0], tf.shape(time_decay)[1], self.hidden_size // self.head_size, self.head_size)), (0,2,3,1)) # time_decay.shape = (batch, head_num, head_size, seq_len)
    time_first = tf.reshape(time_first, (self.hidden_size // self.head_size, self.head_size, 1)) # time_first.shape = (head_num, head_size, 1)
    outs = list()
    for current_index in range(tf.shape(hidden)[1]):
      current_receptance = receptance[...,current_index:current_index+1,:] # current_receptance.shape = (batch, head_num, 1, head_size)
      current_key = key[...,current_index:current_index+1] # current_key.shape = (batch, head_num, head_size, 1)
      current_value = value[...,current_index:current_index+1,:] # current_value.shape = (batch, head_num, 1, head_size)
      current_time_decay = time_decay[...,current_index:current_index+1] # current_time_decay.shape = (batch, head_num, head_size, 1)
      # NOTE: out product of key and value vector
      attention_output = tf.linalg.matmul(current_key, current_value) # attention_output.shape = (batch, head_num, head_size, head_size)
      out = tf.squeeze(tf.linalg.matmul(current_receptance, time_first * attention_output + layer_state), axis = 2) # out.shape = (batch, head num, head size)
      outs.append(out)
    out = tf.stack(outs, axis = 1) # out.shape = (batch, seq_len, head num, head size)
    layer_state = attention_output + current_time_decay * layer_state # layer_state.shape = (batch, head num, head size, head size)
    attn_kv = layer_state
    # 3) output linear
    out = tf.reshape(out, (tf.shape(out)[0], tf.shape(out)[1], self.hidden_size)) # out.shape = (batch, seq_len, hidden_size)
    out = self.group_norm(out) # out.shape = (batch, seq_len, hidden_size)
    out = out * gate
    out = tf.linalg.matmul(out, self.output_w) # out.shape = (batch, seq_len, hidden_size)
    return out, attn_x, attn_kv, ffn_x
  def get_config(self):
    config = super(RWKVAttention, self).get_config()
    config['hidden_size'] = self.hidden_size
    config['head_size'] = self.head_size
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

class RWKVForward(tf.keras.layers.Layer):
  def __init__(self, hidden_size):
    super(RWKVForward, self).__init__()
    self.hidden_size = hidden_size
    self.intermediate_size = int((self.hidden_size * 3.5) // 32 * 32)
  def build(self, input_shape):
    self.time_maa_k = self.add_weight(shape = (1,1,self.hidden_size), dtype = tf.float32, trainable = True, name = 'time_maa_k')
    self.time_maa_r = self.add_weight(shape = (1,1,self.hidden_size), dtype = tf.float32, trainable = True, name = 'time_maa_r')
    self.key_w = self.add_weight(shape = (self.hidden_size, self.intermediate_size), dtype = tf.float32, trainable = True, name = 'key_w')
    self.receptance_w = self.add_weight(shape = (self.hidden_size, self.hidden_size), dtype = tf.float32, trainable = True, name = 'receptance_w')
    self.value_w = self.add_weight(shape = (self.intermediate_size, self.hidden_size), dtype = tf.float32, trainable = True, name = 'value_w')
    super(RWKVForward, self).build(input_shape)
  def compute_output_shape(self, input_shape):
    return input_shape
  def call(self, inputs):
    hidden, attn_x, attn_kv, ffn_x = inputs
    # hidden.shape = (batch, seq_len, hidden)
    # attn_x.shape = (batch, hidden)
    # attn_kv.shape = (batch, hidden // head, head, head)
    # ffn_x.shape = (batch, hidden)
    shifted = tf.concat([tf.expand_dims(ffn_x, axis = 1), hidden[:,0:-1,:]], axis = 1) # shifted.shape = (batch, seq_len, hidden)
    delta_hidden_to_shifted = shifted - hidden
    key = hidden + delta_hidden_to_shifted * self.time_maa_k
    receptance = hidden + delta_hidden_to_shifted * self.time_maa_r

    key = tf.nn.relu(tf.linalg.matmul(key, self.key_w)) ** 2
    value = tf.linalg.matmul(key, self.value_w)
    receptance = tf.math.sigmoid(tf.linalg.matmul(receptance, self.receptance_w))
    
    ffn_x = hidden[:, -1]
    out = receptance * value
    return out, attn_x, attn_kv, ffn_x
  def get_config(self,):
    config = super(RWKVForward,self).get_config()
    config['hidden_size'] = self.hidden_size
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

def RWKVBlock(hidden_size = 768, head_size = 64):
  hidden = tf.keras.Input((None, hidden_size)) # hidden.shape = (batch, seq_len, hidden)
  attn_x = tf.keras.Input((hidden_size,))
  attn_kv = tf.keras.Input((hidden_size // head_size, head_size, head_size,))
  ffn_x = tf.keras.Input((hidden_size,))
  hidden_ = tf.keras.layers.LayerNormalization()(hidden)
  attention, attn_x_, attn_kv_, ffn_x_ = RWKVAttention(hidden_size, head_size)([hidden_, attn_x, attn_kv, ffn_x])
  hidden_ = tf.keras.layers.Add()([hidden_, attention]) # hidden.shape = (batch, seq_len, hidden)
  hidden_ = tf.keras.layers.LayerNormalization()(hidden_)
  feed_forward, attn_x_, attn_kv_, ffn_x_ = RWKVForward(hidden_size)([hidden_, attn_x_, attn_kv_, ffn_x_])
  hidden_ = tf.keras.layers.Add()([hidden_, feed_forward])
  return tf.keras.Model(inputs = (hidden, attn_x, attn_kv, ffn_x), outputs = (hidden_, attn_x_, attn_kv_, ffn_x_))

def RWKV(vocab_size, hidden_size = 768, use_cache = True, num_hidden_layers = 12, head_size = 64):
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
  attn_x_list = list()
  attn_kv_list = list()
  ffn_x_list = list()
  for i in range(num_hidden_layers):
    attn_x = tf.keras.layers.Lambda(lambda x, i: x[...,i], arguments = {'i': i})(state_attn_x)
    attn_kv = tf.keras.layers.Lambda(lambda x, i: x[...,i], arguments = {'i': i})(state_attn_kv)
    ffn_x = tf.keras.layers.Lambda(lambda x, i: x[...,i], arguments = {'i': i})(state_ffn_x)
    hidden_states, attn_x, attn_kv, ffn_x = RWKVBlock(hidden_size, head_size)([hidden_states, attn_x, attn_kv, ffn_x])
    attn_x_list.append(attn_x)
    attn_kv_list.append(attn_kv)
    ffn_x_list.append(ffn_x)
  attn_x = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))(attn_x_list)
  attn_kv = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))(attn_kv_list)
  ffn_x = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))(ffn_x_list)
  hidden_states = tf.keras.layers.LayerNormalization()(hidden_states)
  return tf.keras.Model(inputs = [inputs,] + ([state_attn_x, state_attn_kv, state_ffn_x] if use_cache else []),
                        outputs = [hidden_states, attn_x, attn_kv, ffn_x])

if __name__ == "__main__":
  rwkv = RWKV(vocab_size = 1000)
  hidden = tf.random.uniform(minval = 0, maxval = 1000, shape = (2, 100), dtype = tf.int32)
  attn_x = tf.random.normal(shape = (2, 768, 12), dtype = tf.float32)
  attn_kv = tf.random.normal(shape = (2, 768 // 64, 64, 64, 12), dtype = tf.float32)
  ffn_x = tf.random.normal(shape = (2, 768, 12), dtype = tf.float32)
  outputs, attn_x, attn_kv, ffn_x = rwkv([hidden, attn_x, attn_kv, ffn_x])
  rwkv.save('rwkv.keras')
  print(outputs.shape, attn_x.shape, attn_kv.shape, ffn_x.shape)

