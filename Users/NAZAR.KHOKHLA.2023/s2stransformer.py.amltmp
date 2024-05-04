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

warnings.filterwarnings("ignore")

K = keras.backend
ON_KAGGLE = os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None
FONT_COLOR = "#141B4D"
BACKGROUND_COLOR = "#F6F5F5"
CLR = (Style.BRIGHT + Fore.BLACK) if ON_KAGGLE else (Style.BRIGHT + Fore.WHITE)
RED = Style.BRIGHT + Fore.RED
BLUE = Style.BRIGHT + Fore.BLUE
CYAN = Style.BRIGHT + Fore.CYAN
RESET = Style.RESET_ALL
NOTEBOOK_PALETTE = {
    "DeepPlum": "#3F384A",
    "RubyRed": "#E04C5F",
    "SunburstOrange": "#FFB74D",
}


# def download_dataset_from_kaggle(user, dataset, directory):
#     command = "kaggle datasets download -d "
#     filepath = directory / (dataset + ".zip")

#     if not filepath.is_file():
#         subprocess.run((command + user + "/" + dataset).split())
#         filepath.parent.mkdir(parents=True, exist_ok=True)
#         shutil.unpack_archive(dataset + ".zip", "data")
#         shutil.move(dataset + ".zip", "data")


HTML(
    """
<style>
code {
    background: rgba(42, 53, 125, 0.10) !important;
    border-radius: 4px !important;
}
</style>
"""
)


# In[2]:


easy_dataset_user = "devicharith"
easy_dataset = "language-translation-englishfrench"
easy_dataset_path = Path("combined4.csv")



# In[3]:


easy_dataset = pd.read_csv(easy_dataset_path, encoding="utf-8", engine="pyarrow")
easy_dataset = easy_dataset.sample(len(easy_dataset), random_state=42)
easy_dataset.head()


# In[4]:


easy_dataset.info()


# In[5]:


easy_dataset["English Words in Sentence"] = (
    easy_dataset["English words/sentences"].str.split().apply(len)
)
easy_dataset["Turkish Words in Sentence"] = (
    easy_dataset["Turkish words/sentences"].str.split().apply(len)
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


sentences_en = easy_dataset["English words/sentences"].to_numpy()
sentences_fr = easy_dataset["Turkish words/sentences"].to_numpy()

valid_fraction = 0.1
valid_len = int(valid_fraction * len(easy_dataset))

sentences_en_train = sentences_en[:-valid_len]
sentences_fr_train = sentences_fr[:-valid_len]

sentences_en_valid = sentences_en[-valid_len:]
sentences_fr_valid = sentences_fr[-valid_len:]


# In[7]:


def prepare_input_and_target(sentences_en, sentences_fr):
    """Return data in the format: `((encoder_input, decoder_input), target)`"""
    return (sentences_en, b"startofseq " + sentences_fr), sentences_fr + b" endofseq"


def from_sentences_dataset(
    sentences_en,
    sentences_fr,
    batch_size=32,
    cache=True,
    shuffle=False,
    shuffle_buffer_size=10_000,
    seed=None,
):
    """Creates `TensorFlow` dataset for encoder-decoder RNN from given sentences."""
    dataset = tf.data.Dataset.from_tensor_slices((sentences_en, sentences_fr))
    dataset = dataset.map(prepare_input_and_target, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size)


# In[8]:


benchmark_ds = from_sentences_dataset(sentences_en_train, sentences_fr_train)
benchmark_ds = benchmark_ds.prefetch(tf.data.AUTOTUNE)
bench_results = tfds.benchmark(benchmark_ds, batch_size=32)


# In[9]:


example_ds = from_sentences_dataset(
    sentences_en_train, sentences_fr_train, batch_size=4
)
list(example_ds.take(1))[0]


# In[10]:


example_ds.cardinality()  # Number of batches per epoch.


# In[11]:


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
    model.vectorization_fr.adapt(
        train_dataset.map(
            lambda sentences, target: sentences[1] + b" endofseq",  # French sentences.
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    )

    train_dataset_prepared = train_dataset.map(
        lambda sentences, target: (sentences, model.vectorization_fr(target)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    valid_dataset_prepared = valid_dataset.map(
        lambda sentences, target: (sentences, model.vectorization_fr(target)),
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
        predicted_word = model.vectorization_fr.get_vocabulary()[predicted_word_id]
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
        self.vectorization_fr = layers.TextVectorization(
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
        decoder_input_ids = self.vectorization_fr(decoder_inputs)

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


K.clear_session()  # Resets all state generated by Keras.
tf.random.set_seed(42)  # Ensure reproducibility on CPU.

easy_train_ds = from_sentences_dataset(
    sentences_en_train, sentences_fr_train, shuffle=True, seed=42
)
easy_valid_ds = from_sentences_dataset(sentences_en_valid, sentences_fr_valid)

bidirect_encoder_decoder = BidirectionalEncoderDecoderWithAttention(max_sentence_len=15)


bidirect_history = adapt_compile_and_fit(
    bidirect_encoder_decoder,
    easy_train_ds,
    easy_valid_ds,
    init_lr=0.01,
    lr_decay_rate=0.01,
    colorama_verbose=True,
)


# In[16]:


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
    font_color=FONT_COLOR,
    title_font_size=18,
    plot_bgcolor=BACKGROUND_COLOR,
    paper_bgcolor=BACKGROUND_COLOR,
)
fig.show()



# In[17]:


translation1 = translate(bidirect_encoder_decoder, "Take a seat")
translation2 = translate(bidirect_encoder_decoder, "I wish Tom was here.")
translation3 = translate(bidirect_encoder_decoder, "She ordered him to do it.")

print(CLR + "Actual Possible Translations:")
print(BLUE + "Take a seat".ljust(25), RED + "-> ", BLUE + "Prends place !")
print(
    BLUE + "I wish Tom was here.".ljust(25),
    RED + "-> ",
    BLUE + "J'aimerais que Tom soit là.",
)
print(
    BLUE + "She ordered him to do it.".ljust(25),
    RED + "-> ",
    BLUE + "Elle lui a ordonné de le faire.",
)
print()
print(CLR + "Model Translations:")
print(BLUE + "Take a seat".ljust(25), RED + "-> ", BLUE + translation1)
print(BLUE + "I wish Tom was here.".ljust(25), RED + "-> ", BLUE + translation2)
print(BLUE + "She ordered him to do it.".ljust(25), RED + "-> ", BLUE + translation3)


# In[18]:


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


# In[19]:


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


# In[20]:


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
        self.vectorization_fr = layers.TextVectorization(
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
        decoder_input_ids = self.vectorization_fr(decoder_inputs)

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


# In[21]:


K.clear_session()
tf.random.set_seed(42)

transformer = Transformer(max_sentence_len=15)
transformer_history = adapt_compile_and_fit(
    transformer, easy_train_ds, easy_valid_ds, colorama_verbose=True
)


# In[23]:


fig = px.line(
    transformer_history.history,
    markers=True,
    height=540,
    width=840,
    symbol="variable",
    labels={"variable": "Variable", "value": "Value", "index": "Epoch"},
    title="Combined Dataset - Transformer Training Process",
    color_discrete_sequence=px.colors.diverging.balance_r,
)
fig.update_layout(
    font_color=FONT_COLOR,
    title_font_size=18,
    plot_bgcolor=BACKGROUND_COLOR,
    paper_bgcolor=BACKGROUND_COLOR,
)
fig.show()


# In[27]:


transformer.save_weights('/mnt/c/Users/krzys/Desktop/MircosoftLLM/TFs2s/transformerEnTrW.weights.h5', overwrite = False)


# In[28]:


transformer.save('/mnt/c/Users/krzys/Desktop/MircosoftLLM/TFs2s/transformerEnTr.keras')


# In[25]:


translation1 = translate(transformer, "Take a seat")
translation2 = translate(transformer, "I wish Tom was here.")
translation3 = translate(transformer, "She ordered him to do it.")

print(CLR + "Actual Possible Translations:")
print(BLUE + "Take a seat".ljust(25), RED + "-> ", BLUE + "Prends place !")
print(
    BLUE + "I wish Tom was here.".ljust(25),
    RED + "-> ",
    BLUE + "J'aimerais que Tom soit là.",
)
print(
    BLUE + "She ordered him to do it.".ljust(25),
    RED + "-> ",
    BLUE + "Elle lui a ordonné de le faire.",
)
print()
print(CLR + "Model Translations:")
print(BLUE + "Take a seat".ljust(25), RED + "-> ", BLUE + translation1)
print(BLUE + "I wish Tom was here.".ljust(25), RED + "-> ", BLUE + translation2)
print(BLUE + "She ordered him to do it.".ljust(25), RED + "-> ", BLUE + translation3)

