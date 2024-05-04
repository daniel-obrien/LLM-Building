#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
from colorama import Fore, Style
from IPython.core.display import HTML


# In[2]:


class Transformer(keras.Model):
    def __init__(
        self,
        vocabulary_size=5000,
        max_sentence_len=50,
        embedding_size=256,
        n_encoder_decoder_blocks=1,
        n_attention_heads=8,
        n_units_dense=256,
        dropout_rate=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_sentence_len = max_sentence_len

        self.vectorization_en = layers.TextVectorization(
            vocabulary_size, output_sequence_length=max_sentence_len
        )
        self.vectorization_tr = layers.TextVectorization(
            vocabulary_size, output_sequence_length=max_sentence_len
        )
        self.encoder_embedding = layers.Embedding(
            vocabulary_size, embedding_size, mask_zero=True
        )
        self.decoder_embedding = layers.Embedding(
            vocabulary_size, embedding_size, mask_zero=True
        )
        self.positional_encoding = PositionalEncoding(max_sentence_len, embedding_size)
        self.encoder_blocks = [
            Encoder(embedding_size, n_attention_heads, n_units_dense, dropout_rate)
            for _ in range(n_encoder_decoder_blocks)
        ]
        self.decoder_blocks = [
            Decoder(embedding_size, n_attention_heads, n_units_dense, dropout_rate)
            for _ in range(n_encoder_decoder_blocks)
        ]
        self.output_layer = layers.Dense(vocabulary_size, activation="softmax")

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs

        encoder_input_ids = self.vectorization_en(encoder_inputs)
        decoder_input_ids = self.vectorization_tr(decoder_inputs)

        encoder_embeddings = self.encoder_embedding(encoder_input_ids)
        decoder_embeddings = self.decoder_embedding(decoder_input_ids)

        encoder_pos_embeddings = self.positional_encoding(encoder_embeddings)
        decoder_pos_embeddings = self.positional_encoding(decoder_embeddings)

        encoder_pad_mask = tf.math.not_equal(encoder_input_ids, 0)[:, tf.newaxis]
        decoder_pad_mask = tf.math.not_equal(decoder_input_ids, 0)[:, tf.newaxis]

        # From original paper: "This masking, combined with fact that the output
        # embeddings are offset by one position, ensures that the predictions for
        # position i can depend only on the known outputs at positions less than i."
        batch_max_len_decoder = tf.shape(decoder_embeddings)[1]
        decoder_causal_mask = tf.linalg.band_part(  # Lower triangular matrix.
            tf.ones((batch_max_len_decoder, batch_max_len_decoder), tf.bool), -1, 0
        )
        decoder_mask = decoder_causal_mask & decoder_pad_mask

        Z = encoder_pos_embeddings
        for encoder_block in self.encoder_blocks:
            Z = encoder_block(Z, mask=encoder_pad_mask)

        encoder_output = Z
        Z = decoder_pos_embeddings
        for decoder_block in self.decoder_blocks:
            Z = decoder_block(
                [Z, encoder_output], mask=[decoder_mask, encoder_pad_mask]
            )

        return self.output_layer(Z)


# In[3]:


def adapt_compile(
    model,
    train_dataset,
    valid_dataset,
    n_epochs=25,
    n_patience=5,
    init_lr=0.001,
    lr_decay_rate=0.1,
    colorama_verbose=False,
):
    """Takes the model vectorization layers and adapts them to the training data.
    Then, it prepares the final datasets vectorizing targets and prefetching,
    and finally trains the given model. Additionally, provides learning rate scheduling
    (exponential decay), early stopping and colorama verbose."""

    model.vectorization_en.adapt(
        train_dataset.map(
            lambda sentences, target: sentences[0],  # English sentences.
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    )
    model.vectorization_tr.adapt(
        train_dataset.map(
            lambda sentences, target: sentences[1] + b" endofseq",  # Turkish sentences.
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    )

    train_dataset_prepared = train_dataset.map(
        lambda sentences, target: (sentences, model.vectorization_tr(target)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    valid_dataset_prepared = valid_dataset.map(
        lambda sentences, target: (sentences, model.vectorization_tr(target)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=n_patience, restore_best_weights=True
    )
    
    # The line below doesn't work with multi-file interleaving.
    # n_decay_steps = n_epochs * train_dataset_prepared.cardinality().numpy()
    # Less elegant solution.
    n_decay_steps = n_epochs * len(list(train_dataset_prepared))
    scheduled_lr = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init_lr,
        decay_steps=n_decay_steps,
        decay_rate=lr_decay_rate,
    )

    model_callbacks = [early_stopping_cb]
    verbose_level = 1
    if colorama_verbose:
        model_callbacks.append(ColoramaVerbose())
        verbose_level = 0

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.RMSprop(learning_rate=scheduled_lr),
        metrics=["accuracy"],
    )

    return model


# In[4]:


easy_dataset_path = Path("/home/azureuser/cloudfiles/code/Users/NAZAR.KHOKHLA.2023/combined4.csv")


# In[5]:


easy_dataset = pd.read_csv(easy_dataset_path, encoding="utf-8", engine="pyarrow")
easy_dataset = easy_dataset.sample(len(easy_dataset), random_state=42)
easy_dataset.head()


# In[6]:


sentences_en = easy_dataset["English"].to_numpy()
sentences_tr = easy_dataset["The Other Language"].to_numpy()

valid_fraction = 0.1
valid_len = int(valid_fraction * len(easy_dataset))

sentences_en_train = sentences_en[:-valid_len]
sentences_tr_train = sentences_tr[:-valid_len]

sentences_en_valid = sentences_en[-valid_len:]
sentences_tr_valid = sentences_tr[-valid_len:]


# In[7]:


def prepare_input_and_target(sentences_en, sentences_tr):
    """Return data in the format: `((encoder_input, decoder_input), target)`"""
    return (sentences_en, b"startofseq " + sentences_tr), sentences_tr + b" endofseq"


def from_sentences_dataset(
    sentences_en,
    sentences_tr,
    batch_size=32,
    cache=True,
    shuffle=False,
    shuffle_buffer_size=10_000,
    seed=None,
):
    """Creates `TensorFlow` dataset for encoder-decoder RNN from given sentences."""
    dataset = tf.data.Dataset.from_tensor_slices((sentences_en, sentences_tr))
    dataset = dataset.map(prepare_input_and_target, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size)


# In[8]:


keras.backend.clear_session()
tf.random.set_seed(42)

easy_train_ds = from_sentences_dataset(
    sentences_en_train, sentences_tr_train, shuffle=True, seed=42
)
easy_valid_ds = from_sentences_dataset(sentences_en_valid, sentences_tr_valid)


# In[9]:


class PositionalEncoding(layers.Layer):
    def __init__(
        self, max_sentence_len=50, embedding_size=256, dtype=tf.float32, **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        if not embedding_size % 2 == 0:
            raise ValueError("The `embedding_size` must be even.")

        p, i = np.meshgrid(np.arange(max_sentence_len), np.arange(embedding_size // 2))
        pos_emb = np.empty((1, max_sentence_len, embedding_size))
        pos_emb[:, :, 0::2] = np.sin(p / 10_000 ** (2 * i / embedding_size)).T
        pos_emb[:, :, 1::2] = np.cos(p / 10_000 ** (2 * i / embedding_size)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
        self.supports_masking = True

    def call(self, inputs):
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.positional_embedding[:, :batch_max_length]


# In[10]:


class Encoder(layers.Layer):
    def __init__(
        self,
        embedding_size=256,
        n_attention_heads=8,
        n_units_dense=256,
        dropout_rate=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.multi_head_attention = layers.MultiHeadAttention(
            n_attention_heads, embedding_size, dropout=dropout_rate
        )
        self.feed_forward = keras.Sequential(
            [
                layers.Dense(
                    n_units_dense, activation="relu", kernel_initializer="he_normal"
                ),
                layers.Dense(embedding_size, kernel_initializer="he_normal"),
                layers.Dropout(dropout_rate),
            ]
        )
        self.add = layers.Add()
        self.normalization = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        Z = inputs
        skip_Z = Z
        Z = self.multi_head_attention(Z, value=Z, attention_mask=mask)
        Z = self.normalization(self.add([Z, skip_Z]))
        skip_Z = Z
        Z = self.feed_forward(Z)
        return self.normalization(self.add([Z, skip_Z]))


class Decoder(layers.Layer):
    def __init__(
        self,
        embedding_size=256,
        n_attention_heads=8,
        n_units_dense=256,
        dropout_rate=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.masked_multi_head_attention = layers.MultiHeadAttention(
            n_attention_heads, embedding_size, dropout=dropout_rate
        )
        self.multi_head_attention = layers.MultiHeadAttention(
            n_attention_heads, embedding_size, dropout=dropout_rate
        )
        self.feed_forward = keras.Sequential(
            [
                layers.Dense(
                    n_units_dense, activation="relu", kernel_initializer="he_normal"
                ),
                layers.Dense(embedding_size, kernel_initializer="he_normal"),
                layers.Dropout(dropout_rate),
            ]
        )
        self.add = layers.Add()
        self.normalization = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        decoder_mask, encoder_mask = mask  # type: ignore
        Z, encoder_output = inputs
        Z_skip = Z
        Z = self.masked_multi_head_attention(Z, value=Z, attention_mask=decoder_mask)
        Z = self.normalization(self.add([Z, Z_skip]))
        Z_skip = Z
        Z = self.multi_head_attention(
            Z, value=encoder_output, attention_mask=encoder_mask
        )
        Z = self.normalization(self.add([Z, Z_skip]))
        Z_skip = Z
        Z = self.feed_forward(Z)
        return self.normalization(self.add([Z, Z_skip]))


# In[11]:


CLR = (Style.BRIGHT + Fore.WHITE)
RED = Style.BRIGHT + Fore.RED
class ColoramaVerbose(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"{CLR}Epoch: {RED}{epoch + 1:02d}{CLR} -",
            f"{CLR}loss: {RED}{logs['loss']:.5f}{CLR} -",
            f"{CLR}accuracy: {RED}{logs['accuracy']:.5f}{CLR} -",
            f"{CLR}val_loss: {RED}{logs['val_loss']:.5f}{CLR} -",
            f"{CLR}val_accuracy: {RED}{logs['val_accuracy']:.5f}",
        )


# In[18]:


transformer_new = Transformer(max_sentence_len=15)
transformer_new_history = adapt_compile(
    transformer_new, easy_train_ds, easy_valid_ds, colorama_verbose=True
)


# In[19]:


transformer_new.load_weights('model_weights')


# In[20]:


def translate(model, sentence_en):
    translation = ""
    for word_idx in range(model.max_sentence_len):
        X_encoder = np.array([sentence_en])
        X_decoder = np.array(["startofseq " + translation])
        # Last token's probas.
        y_proba = model.predict((X_encoder, X_decoder), verbose=0)[0, word_idx]
        predicted_word_id = np.argmax(y_proba)
        predicted_word = model.vectorization_tr.get_vocabulary()[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()


# In[30]:


translation4 = translate(transformer_new_history, "She ordered him to do it")


# In[31]:


print(translation4)

