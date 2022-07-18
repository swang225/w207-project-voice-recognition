import os
from urllib import request
from tqdm import tqdm
import ssl
import tarfile
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from jiwer import wer

def download_file(url, name, save_path):
    local_filename = os.path.join(save_path, name)
    full_url = os.path.join(url, name)

    gcontext = ssl.SSLContext()
    response = request.urlopen(
        full_url,
        context=gcontext
    )

    with tqdm.wrapattr(
            open(local_filename, "wb"),
            "write",
            miniters=1,
            desc=name,
            total=getattr(response, 'length', None)
    ) as fout:
        for chunk in response:
            fout.write(chunk)


def extract(
        file_path,
        mode="r:bz2"
):
    tar = tarfile.open(file_path, mode=mode)
    tar.extractall()
    tar.close()



def read_metadata(metadata_path, frac=1):
    """
    frac: is the fraction of data you want to use. 1 means 100%, or all data
    """
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]]
    metadata_df = metadata_df.sample(frac=frac).reset_index(drop=True)

    return metadata_df


def load_split_data(metadata_path, frac=1, split=0.9):
    """
    split: is the fraction of data you want to use for training. 0.9 means 90%, rest is for test
    """
    metadata_df = read_metadata(metadata_path, frac=frac)

    split = int(len(metadata_df) * split)
    df_train = metadata_df[:split]
    df_val = metadata_df[split:]

    return df_train, df_val


class LabelEncoder:

    def __init__(self):
        characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
        # encoder
        self.char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
        # decoder
        self.num_to_char = keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True
        )

    def encode(self, label):
        label = tf.strings.lower(label)
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        return self.char_to_num(label)

    def decode(self, nums):
        return tf.strings.reduce_join(self.num_to_char(nums))


def wav_to_audio(filepath):
    file = tf.io.read_file(filepath)

    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)

    audio = tf.cast(audio, tf.float32)

    return audio


def audio_to_spectrogram(
        audio,
        frame_length,
        frame_step,
        fft_length,
):
    spectrogram = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )

    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    return spectrogram


def wav_to_features(
        wav_file,
        wavs_path,
        frame_length,
        frame_step,
        fft_length,
):
    audio = wav_to_audio(wavs_path + wav_file + ".wav")
    spectrogram = audio_to_spectrogram(
            audio=audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
    )

    return spectrogram


class FeatureEncoder:

    def __init__(
            self,
            wavs_path,
            frame_length,
            frame_step,
            fft_length,
    ):
        self.wavs_path = wavs_path
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

    def features(
            self,
            wav_file,
    ):
        return wav_to_features(
            wav_file=wav_file,
            wavs_path=self.wavs_path,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
        )


class AudioContext:

    _instance = None
    _allowed = False

    def __init__(
            self,
            wavs_path,
            frame_length,
            frame_step,
            fft_length,
    ):
        if not AudioContext._allowed:
            raise ValueError("Cannot instantiate AudioContext")

        self.label_encoder = LabelEncoder()
        self.feature_encoder = FeatureEncoder(
            wavs_path=wavs_path,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length
        )

    @staticmethod
    def set(
            wavs_path,
            frame_length,
            frame_step,
            fft_length,
    ):
        AudioContext._allowed = True
        AudioContext._instance = AudioContext(
            wavs_path=wavs_path,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length
        )
        AudioContext._allowed = False

    @staticmethod
    def get():
        if AudioContext._instance is None:
            raise ValueError("AudioContext not set yet")

        return AudioContext._instance


def encode_single_sample(wav_file, label):

    lencoder = AudioContext.get().label_encoder
    fencoder = AudioContext.get().feature_encoder

    features = fencoder.features(wav_file)
    encoded_label = lencoder.encode(label)

    return features, encoded_label


def create_dataset(data_df, batch_size):
    _dataset = tf.data.Dataset.from_tensor_slices(
        (
            list(data_df["file_name"]),
            list(data_df["normalized_transcription"])
        )
    )
    _dataset = (
        _dataset.map(
            encode_single_sample,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return _dataset


def CTCLoss(
        y_true,
        y_pred
):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )[0][0]

    # Iterate over the results and get back the text
    lencoder = AudioContext.get().label_encoder
    res = []
    for result in results:
        result = lencoder.decode(result).numpy().decode("utf-8")
        res.append(result)
    return res


class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset, model):
        super().__init__()
        self.dataset = dataset
        self.model = model

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        lencoder = AudioContext.get().label_encoder
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    lencoder.decode(label).numpy().decode("utf-8")\
                )
                targets.append(label)

        wer_score = wer(targets, predictions)

        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)


def build_model001(
        input_dim,
        output_dim,
        filter_size1=32,
        filter_size2=32,
        kernel_size1=[11, 41],
        kernel_size2=[11, 21],
        strides1=[2, 2],
        strides2=[1, 2],
        rnn_layers=1,
        rnn_units=128,
        drop1=0.5,
        drop2=0.5,
):
    # Model's input
    input_spectrogram = layers.Input(
        (None, input_dim),
        name="input"
    )

    # Expand the dimension to use 2D CNN.
    x = layers.Reshape(
        (-1, input_dim, 1),
        name="expand_dim")(input_spectrogram)

    # Convolution layer 1
    x = layers.Conv2D(
        filters=filter_size1,
        kernel_size=kernel_size1,
        strides=strides1,
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)

    # Convolution layer 2
    x = layers.Conv2D(
        filters=filter_size2,
        kernel_size=kernel_size2,
        strides=strides2,
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)

    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape(
        (-1, x.shape[-2] * x.shape[-1])
    )(x)

    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=drop1)(x)

    # Dense layer
    x = layers.Dense(
        units=rnn_units * 2,
        name="dense_1"
    )(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=drop2)(x)

    # Classification layer
    output = layers.Dense(
        units=output_dim + 1,
        activation="softmax"
    )(x)

    # Model
    model = keras.Model(
        input_spectrogram,
        output,
        name="DeepSpeech_2"
    )

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)

    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)

    return model


# params
data_name = "LJSpeech-1.1.tar.bz2"
data_url = "https://data.keithito.com/data/speech/"
save_path = "C:/data"
frame_length = 256
frame_step = 160
fft_length = 384
epochs = 1
frac = 0.1 # using 10% of the available data as input
split = 0.9 # using 90% of the input data to train, 10% for validation

wavs_path = save_path + "/LJSpeech-1.1/wavs/"
metadata_path = save_path + "/LJSpeech-1.1/metadata.csv"

# download and extract lj speech data
download_file(
    url=data_url,
    name=data_name,
    save_path=save_path
)
extract(os.path.join(save_path, data_name))

# Read metadata file and parse it
df_train, df_val = load_split_data(metadata_path, frac=0.3, split=0.9)
print(f"Size of the training set: {len(df_train)}")
print(f"Size of the training set: {len(df_val)}")


AudioContext.set(
    wavs_path=wavs_path,
    frame_length=frame_length,
    frame_step=frame_step,
    fft_length=fft_length,
)
lencoder = AudioContext.get().label_encoder
fencoder = AudioContext.get().feature_encoder

meta_df = read_metadata(metadata_path)
meta_row = meta_df.loc[5000]
print(meta_row.file_name)
print(meta_row.normalized_transcription)

features, encoded_label = encode_single_sample(
    wav_file=meta_row.file_name,
    label=meta_row.normalized_transcription
)
decoded_label = lencoder.decode(encoded_label)
print(features)
print(encoded_label)
print(decoded_label)


batch_size = 32

train_dataset = create_dataset(data_df=df_train, batch_size=batch_size)
validation_dataset = create_dataset(data_df=df_val, batch_size=batch_size)

model001 = build_model001(
    input_dim=fft_length // 2 + 1,
    output_dim=lencoder.char_to_num.vocabulary_size(),
    rnn_units=512,
)
model001.summary(line_length=110)

validation_callback = CallbackEval(
    dataset=validation_dataset,
    model=model001
)

# Train the model
history = model001.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[validation_callback],
)
