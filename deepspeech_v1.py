import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.python.ops import gen_audio_ops as contrib_audio
import numpy as np


def wav_to_audio(
        wav_filename,
):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)

    audio = decoded.audio  # equivalent to frames converted to int16
    sample_rate = int(decoded.sample_rate)

    return audio, sample_rate


def wav_to_spectrogram(
        audio,
        sample_rate,
        feature_win_len=32,  # feature extraction audio window length in milliseconds
        feature_win_step=20  # feature extraction window step length in milliseconds
):
    audio_window_samples = sample_rate * (feature_win_len / 1000)
    audio_step_samples = sample_rate * (feature_win_step / 1000)

    spectrogram = contrib_audio.audio_spectrogram(
        audio,
        window_size=audio_window_samples,
        stride=audio_step_samples,
        magnitude_squared=True
    )

    return spectrogram, sample_rate


def spectrogram_to_features(
        spectrogram,
        sample_rate,
        n_input=26,  # Number of MFCC features, can be determined from sample rate
):
    features = contrib_audio.mfcc(
        spectrogram=spectrogram,
        sample_rate=sample_rate,
        dct_coefficient_count=n_input,
        upper_frequency_limit=sample_rate / 2
    )
    features = tf.reshape(features, [-1, n_input])

    return features


def wav_to_features(
        wav_filename,
        feature_win_len=32,
        feature_win_step=20,
        n_input=26,
):
    audio, sample_rate = wav_to_audio(wav_filename)

    spectrogram, sample_rate = wav_to_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        feature_win_len=feature_win_len,
        feature_win_step=feature_win_step,
    )

    features = spectrogram_to_features(
        spectrogram=spectrogram,
        sample_rate=sample_rate,
        n_input=n_input,
    )

    return features, spectrogram, audio


class LabelEncoder:

    cmap = {
        ' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4,
        'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
        'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14,
        'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
        't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24,
        'y': 25, 'z': 26, '\'': 27,
    }

    @staticmethod
    def to_int(c):
        return LabelEncoder.cmap[c]

    @staticmethod
    def encode(input_string):
        return list(map(LabelEncoder.to_int, input_string))


class OverlapWindow(tf.keras.layers.Layer):
    def __init__(
            self,
            window_width,
            num_channels,
            out_channels
    ):
        super(OverlapWindow, self).__init__()
        eye = np.eye(out_channels)
        eye = eye.reshape(
            window_width,
            num_channels,
            out_channels
        )
        self.eye_filter = tf.constant(eye, tf.float32)
        self.n_input = num_channels
        self.num_channels = num_channels
        self.window_width = window_width

    def call(self, inputs):
        inputs = tf.reshape([inputs], [1, -1, self.n_input])
        inputs = tf.nn.conv1d(
            input=inputs,
            filters=self.eye_filter,
            stride=1,
            padding='SAME'
        )

        inputs = tf.reshape(
            inputs,
            [1, -1, self.window_width, self.num_channels]
        )

        # in keras, one item is processed at a time
        inputs = inputs[0]

        return inputs


def to_sparse_tuple(sequence):
    indices = np.asarray(
        list(zip([0]*len(sequence),
                 range(len(sequence)))),
        dtype=np.int64)

    shape = np.asarray(
        [1, len(sequence)],
        dtype=np.int64
    )
    return indices, sequence, shape


def CTCLoss(y_true, y_pred):
    sequence_length = [tf.shape(y_pred)[0].numpy()]
    y_true = y_true[0]
    y_pred = tf.reshape(y_pred, [y_pred.shape[0], 1, y_pred.shape[1]])
    total_loss = tfv1.nn.ctc_loss(
        labels=y_true,
        inputs=y_pred,
        sequence_length=sequence_length
    )
    return total_loss


n_alphabet = 28
n_input = 26 # number of features
n_context = 9
layer_1_units = 2048
dropout_1_rate = 0.05
layer_2_units = 2048
dropout_2_rate = 0.05
layer_3_units = 2048
dropout_3_rate = 0.05
n_cell_dim = 2048
layer_5_units = 2048
dropout_5_rate = 0.05


import pandas as pd
import os
train_file = "C:/Users/aphri/pycharm/t0001/data/deepspeech/meta/train.csv"
dir_name = "C:/Users/aphri/pycharm/t0001/data/deepspeech/wav"

train_df = pd.read_csv(train_file)

train_x = []
train_y = []
count = 10
for idx, row in train_df.iterrows():
    if count <= 0:
        break
    count -= 1

    features, spectrogram, audio = wav_to_features(os.path.join(dir_name, row.wav_filename), n_input=n_input)
    seq_len = features.shape[0]
    features = tf.pad(features, tf.constant([[0, 1000 - seq_len], [0, 0]]), "CONSTANT")
    train_x.append(features)

    label = tf.SparseTensor(*to_sparse_tuple(LabelEncoder.encode(row.transcript)))
    train_y.append(label)
    print(features.shape)
    print(label.shape)


layer_6_units = n_alphabet + 1 # 1 for ctc blank
window_width = 2 * n_context + 1
num_channels = n_input
out_channels = window_width * num_channels


layer_0 = OverlapWindow(
    window_width=window_width,
    num_channels=num_channels,
    out_channels=out_channels
)


layer_1 = tf.keras.layers.Dense(
      units=layer_1_units,
      use_bias=True,
      activation="relu"
)

dropout_1 = tf.keras.layers.Dropout(rate=dropout_1_rate)

layer_2 = tf.keras.layers.Dense(
      units=layer_2_units,
      use_bias=True,
      activation="relu"
)
dropout_2 = tf.keras.layers.Dropout(rate=dropout_2_rate)

layer_3 = tf.keras.layers.Dense(
      units=layer_3_units,
      use_bias=True,
      activation="relu"
)
dropout_3 = tf.keras.layers.Dropout(rate=dropout_3_rate)

# RNN
fw_cell = tf.keras.layers.LSTM(units=n_cell_dim, name='lstm_cell')

layer_5 = tf.keras.layers.Dense(
      units=layer_5_units,
      use_bias=True,
      activation="relu"
)
dropout_5 = tf.keras.layers.Dropout(rate=dropout_5_rate)

layer_6 = tf.keras.layers.Dense(
      units=layer_6_units,
      use_bias=True,
      activation="relu"
)

model = tf.keras.Sequential(
    [
        layer_0,
        layer_1,
        dropout_1,
        layer_2,
        dropout_2,
        layer_3,
        dropout_3,
        fw_cell,
        layer_5,
        dropout_5,
        layer_6,
    ]
)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss=CTCLoss)

model.fit(train_x, train_y)
