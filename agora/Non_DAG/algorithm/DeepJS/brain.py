import tensorflow as tf
import torch as torch
from torch import nn
import torch.nn.functional as F


class BrainBig(tf.keras.Model):
    name = 'BrainBig'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_6 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        state = self.dense_6(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)


class Brain(tf.keras.Model):
    name = 'Brain'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)


class BrainSmall(tf.keras.Model):
    name = 'BrainSmall'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(6, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)


class BrainTorch(nn.Module):
    def __innit__(self, state_size):
        super().__innit__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def call(self, state):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        state = self.layer3(state)
        return state

    # pair_index = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=1).numpy()[0]
