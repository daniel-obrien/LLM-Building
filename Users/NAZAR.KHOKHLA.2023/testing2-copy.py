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


easy_dataset_path = Path("/home/azureuser/cloudfiles/code/Users/NAZAR.KHOKHLA.2023/sentences.csv")


# In[3]:


easy_dataset = pd.read_csv(easy_dataset_path, encoding="utf-8", engine="pyarrow")
easy_dataset = easy_dataset.sample(len(easy_dataset), random_state=42)
easy_dataset.head()


# In[4]:


easy_dataset.info()


# In[5]:


easy_dataset["English Words in Sentence"] = (
    easy_dataset["English"].str.split().apply(len)
)
easy_dataset["Turkish Words in Sentence"] = (
    easy_dataset["The Other Language"].str.split().apply(len)
)

fig = px.histogram(
    easy_dataset,
    x=["English Words in Sentence", "Turkish Words in Sentence"],
    color_discrete_sequence=["#3f384a", "#e04c5f"],
    labels={"variable": "Variable", "value": "Words in Sentence"},
    marginal="box",
    barmode="group",
    height=540,
    width=840,
    title="Easy Dataset - Words in Sentence",
)
fig.update_layout(
    font_color="#141B4D",
    title_font_size=18,
    plot_bgcolor="#F6F5F5",
    paper_bgcolor="#F6F5F5",
    bargap=0.2,
    bargroupgap=0.1,
    legend=dict(orientation="h", yanchor="bottom", xanchor="right", y=1.02, x=1),
    yaxis_title="Count",
)
fig.show()


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


benchmark_ds = from_sentences_dataset(sentences_en_train, sentences_tr_train)
benchmark_ds = benchmark_ds.prefetch(tf.data.AUTOTUNE)
bench_results = tfds.benchmark(benchmark_ds, batch_size=32)


# In[9]:


example_ds = from_sentences_dataset(
    sentences_en_train, sentences_tr_train, batch_size=4
)
list(example_ds.take(1))[0]


# In[10]:


example_ds.cardinality()  # Number of batches per epoch.


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


# In[12]:


def adapt_compile_and_fit(
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

    return model.fit(
        train_dataset_prepared,
        epochs=n_epochs,
        validation_data=valid_dataset_prepared,
        callbacks=model_callbacks,
        verbose=verbose_level,
    )


# In[13]:


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


# In[14]:


class BidirectionalEncoderDecoderWithAttention(keras.Model):
    def __init__(
        self,
        vocabulary_size=5000,
        max_sentence_len=50,
        embedding_size=256,
        n_units_lstm=512,
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

        self.encoder = layers.Bidirectional(
            layers.LSTM(n_units_lstm // 2, return_sequences=True, return_state=True)
        )
        self.decoder = layers.LSTM(n_units_lstm, return_sequences=True)
        self.attention = layers.Attention()
        self.output_layer = layers.Dense(vocabulary_size, activation="softmax")

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs

        encoder_input_ids = self.vectorization_en(encoder_inputs)
        decoder_input_ids = self.vectorization_tr(decoder_inputs)

        encoder_embeddings = self.encoder_embedding(encoder_input_ids)
        decoder_embeddings = self.decoder_embedding(decoder_input_ids)

        # The final hidden state of the encoder, representing the entire
        # input sequence, is used to initialize the decoder.
        encoder_output, *encoder_state = self.encoder(encoder_embeddings)
        encoder_state = [
            tf.concat(encoder_state[0::2], axis=-1),  # Short-term state (0 & 2).
            tf.concat(encoder_state[1::2], axis=-1),  # Long-term state (1 & 3).
        ]
        decoder_output = self.decoder(decoder_embeddings, initial_state=encoder_state)
        attention_output = self.attention([decoder_output, encoder_output])

        return self.output_layer(attention_output)


# In[15]:


keras.backend.clear_session()  # Resets all state generated by Keras.
tf.random.set_seed(42)  # Ensure reproducibility on CPU.

easy_train_ds = from_sentences_dataset(
    sentences_en_train, sentences_tr_train, shuffle=True, seed=42
)
easy_valid_ds = from_sentences_dataset(sentences_en_valid, sentences_tr_valid)

bidirect_encoder_decoder = BidirectionalEncoderDecoderWithAttention(max_sentence_len=15)
bidirect_history = adapt_compile_and_fit(
    bidirect_encoder_decoder,
    easy_train_ds,
    easy_valid_ds,
    init_lr=0.01,
    lr_decay_rate=0.01,
    colorama_verbose=True,
)


# In[18]:


fig = px.line(
    bidirect_history.history,
    markers=True,
    height=540,
    width=840,
    symbol="variable",
    labels={"variable": "Variable", "value": "Value", "index": "Epoch"},
    title="Easy Dataset - Encoder-Decoder RNN Training Process",
    color_discrete_sequence=px.colors.diverging.balance_r,
)
fig.update_layout(
    font_color="#141B4D",
    title_font_size=18,
    plot_bgcolor="#F6F5F5",
    paper_bgcolor="#F6F5F5",
)
fig.show()


# In[23]:


translation1 = translate(bidirect_encoder_decoder, "Tell Tom to take a seat.")
translation2 = translate(bidirect_encoder_decoder, "I wish Tom were dead.")
translation3 = translate(bidirect_encoder_decoder, "She ordered three dinners.")

print("Tell Tom to take a seat.     Tom'a oturmasını söyle.")
print("I wish Tom were dead.        Keşke Tom ölmüş olsa.")
print("She ordered three dinners.   O üç tane akşam yemeği sipariş etti.")
print()
print("Model Translations:")
print(translation1)
print(translation2)
print(translation3)


# In[20]:


import mlflow
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env


# In[21]:


loss_history =bidirect_history.history['loss']


# In[26]:


# Assuming `train_dataset` and `bidirect_encoder_decoder` are available
model = bidirect_encoder_decoder
train_dataset = easy_train_ds

# Adapt the vectorization layers
model.vectorization_en.adapt(
    train_dataset.map(
        lambda sentences, target: sentences[0],  # English sentences.
        num_parallel_calls=tf.data.AUTOTUNE,
    )
)


# In[27]:


model.vectorization_tr.adapt(
    train_dataset.map(
        lambda sentences, target: sentences[1] + b" endofseq",  # Turkish sentences.
        num_parallel_calls=tf.data.AUTOTUNE,
    )
)


# In[28]:


train_dataset_prepared = train_dataset.map(
    lambda sentences, target: (sentences, model.vectorization_tr(target)),
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)


# In[ ]:





# In[22]:


mlflow.set_experiment("Your-First-AI")

with mlflow.start_run( run_name='run1'):
    for loss in loss_history:
        mlflow.log_metric("loss", loss)
    signature = infer_signature(np.array(["Tell Tom to take a seat."]), translate(bidirect_encoder_decoder, "Tell Tom to take a seat."))
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=None,
        additional_conda_channels=None
    )
mlflow.tensorflow.log_model(bidirect_encoder_decoder, "bidirect_encoder_decoder", conda_env=conda_env, signature=signature)

