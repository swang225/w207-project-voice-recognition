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
import pickle
from os.path import exists


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


def load_split_data_alexa(metadata_path, frac=1, split1=0.8, split2=0.5):
    """
    split: is the fraction of data you want to use for training. 0.9 means 90%, rest is for test
    """

    all_files = []
    for subdir, dirs, files in os.walk(metadata_path):
        for file in files:
            full_path = os.path.join(subdir, file)
            all_files.append(full_path)

    df = pd.DataFrame(data={"file_name": all_files})
    df["normalized_transcription"] = "alexa"
    metadata_df = df
    metadata_df = metadata_df.sample(frac=frac).reset_index(drop=True)

    split1 = int(len(metadata_df) * split1)
    df_train = metadata_df[:split1]
    df_val = metadata_df[split1:]

    split2 = int(len(df_val) * split2)
    df_test = df_val[split2:]
    df_val = df_val[:split2]

    return df_train, df_val, df_test


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
        add_suffix=True,
):
    file_path = wavs_path + wav_file
    if add_suffix:
        file_path += ".wav"
    audio = wav_to_audio(file_path)
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
            add_suffix,
    ):
        self.wavs_path = wavs_path
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.add_suffix = add_suffix

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
            add_suffix=self.add_suffix
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
            add_suffix,
    ):
        if not AudioContext._allowed:
            raise ValueError("Cannot instantiate AudioContext")

        self.label_encoder = LabelEncoder()
        self.feature_encoder = FeatureEncoder(
            wavs_path=wavs_path,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            add_suffix=add_suffix,
        )

    @staticmethod
    def set(
            wavs_path,
            frame_length,
            frame_step,
            fft_length,
            add_suffix,
    ):
        AudioContext._allowed = True
        AudioContext._instance = AudioContext(
            wavs_path=wavs_path,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            add_suffix=add_suffix,
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


def decode_prediction(pred):
    pred = tf.reshape(pred, [1, pred.shape[0], pred.shape[1]])
    return decode_batch_predictions(pred)


def accuracy(labels, predictions):

    def _acc(l, p):

        l_len = len(l)
        p_len = len(p)
        count = 0
        for i in range(min(l_len, p_len)):
            if l[i] == p[i]:
                count += 1

        return count / max(l_len, p_len)

    total_acc = 0
    for l, p in zip(labels, predictions):
        total_acc += _acc(l, p)

    avg_acc = total_acc / len(labels)
    return avg_acc


def abs_accuracy(labels, predictions):

    def _acc(l, p):

        if l == p:
            return 1

        return 0

    total_acc = 0
    for l, p in zip(labels, predictions):
        total_acc += _acc(l, p)

    avg_acc = total_acc / len(labels)
    return avg_acc


class ModelEvaluator(keras.callbacks.Callback):

    def __init__(self, dataset, model):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.history = []

    @staticmethod
    def print(wer_score, res_acc, abs_acc, pred_df):
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        print(f"Accuracy: {res_acc:.4f}")
        print("-" * 100)
        print(f"Abs accuracy: {abs_acc:.4f}")
        print("-" * 100)
        print(pred_df.sample(n=5))

    def do_prediction(self):
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
                    lencoder.decode(label).numpy().decode("utf-8") \
                    )
                targets.append(label)

        wer_score = wer(targets, predictions)
        res_acc = accuracy(targets, predictions)
        abs_acc = abs_accuracy(targets, predictions)
        pred_df = pd.DataFrame(data={"Label": targets, "predictions": predictions})

        return wer_score, res_acc, abs_acc, pred_df

    def on_epoch_end(self, epoch: int, logs=None):
        wer_score, res_acc, abs_acc, pred_df = self.do_prediction()
        self.history.append((wer_score, res_acc, abs_acc, pred_df))
        self.print(wer_score, res_acc, abs_acc, pred_df)


def evaluate(model, test_dataset):
    print("evaluate model against test dataset")
    test_evaluator = ModelEvaluator(
        dataset=test_dataset,
        model=model
    )
    wer_score, res_acc, abs_acc, pred_df = test_evaluator.do_prediction()
    test_evaluator.print(wer_score, res_acc, abs_acc, pred_df)
    return wer_score, res_acc, abs_acc, pred_df


def write_pickle(data, file, ensure_exist=True):

    if ensure_exist:
        path = os.path.dirname(file)
        if len(path) > 0:
            os.makedirs(path, exist_ok=True)

    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(file):
    with open(file, 'rb') as handle:
        b = pickle.load(handle)

    return b


def build_model_rnn(
        input_dim,
        output_dim,
        model_name,
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
        name=f"{model_name}_input"
    )

    # Expand the dimension to use 2D CNN.
    x = layers.Reshape(
        (-1, input_dim, 1),
        name=f"{model_name}_expand_dim")(input_spectrogram)

    # Convolution layer 1
    x = layers.Conv2D(
        filters=filter_size1,
        kernel_size=kernel_size1,
        strides=strides1,
        padding="same",
        use_bias=False,
        name=f"{model_name}_conv_1",
    )(x)
    x = layers.BatchNormalization(name=f"{model_name}_conv_1_bn")(x)
    x = layers.ReLU(name=f"{model_name}_conv_1_relu")(x)

    # Convolution layer 2
    x = layers.Conv2D(
        filters=filter_size2,
        kernel_size=kernel_size2,
        strides=strides2,
        padding="same",
        use_bias=False,
        name=f"{model_name}_conv_2",
    )(x)
    x = layers.BatchNormalization(name=f"{model_name}_conv_2_bn")(x)
    x = layers.ReLU(name=f"{model_name}_conv_2_relu")(x)

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
            name=f"{model_name}_gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"{model_name}_bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=drop1)(x)

    # Dense layer
    x = layers.Dense(
        units=rnn_units * 2,
        name=f"{model_name}_dense_1"
    )(x)
    x = layers.ReLU(name=f"{model_name}_dense_1_relu")(x)
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
        name=f"{model_name}_DeepSpeech_2"
    )

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)

    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)

    return model


def build_model_simple_logistic(
        input_dim,
        output_dim,
        model_name,
):
    # Model's input
    input_spectrogram = layers.Input(
        (None, input_dim),
        name=f"{model_name}_input"
    )

    # Classification layer
    output = layers.Dense(
        units=output_dim + 1,
        activation="softmax"
    )(input_spectrogram)

    # Model
    model = keras.Model(
        input_spectrogram,
        output,
        name=f"{model_name}_SimpleLogistic"
    )

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)

    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)

    return model


def train_model(model_config, save_path, check_exist=True):
    model_name = model_config[0]
    if check_exist and exists(os.path.join(save_path, model_name)):
        print(f"model {model_name} already exists")
        return

    model_kwargs = model_config[1]
    model_builder = model_config[2]

    model = model_builder(
        input_dim=fft_length // 2 + 1,
        output_dim=lencoder.char_to_num.vocabulary_size(),
        model_name=model_name,
        **model_kwargs,
    )
    model.summary(line_length=110)

    # create validation evaluator
    validation_callback = ModelEvaluator(
        dataset=validation_dataset,
        model=model
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[validation_callback],
    )

    # save the model
    model.save(os.path.join(save_path, model_name))
    write_pickle(
        {
            "validation results": validation_callback.history,
            "train history": history.history
        },
        os.path.join(save_path, f"{model_name}_validation.pkl")
    )


def load_model(model_name, save_path):

    model = keras.models.load_model(
        os.path.join(save_path, model_name),
        custom_objects={"CTCLoss": CTCLoss})
    model_res = read_pickle(os.path.join(save_path, f"{model_name}_validation.pkl"))

    return model, model_res


# params
frame_length = 256
frame_step = 160
fft_length = 384
epochs = 10
frac = 1
split1 = 0.8 # using 80% of the input data to train, 20% for validation
split2 = 0.5 # using 50% of validation for model tuning, 50% for testing
save_path = "C:/data"

metadata_path = "C:/Users/aphri/pycharm/t0001/data/alexa1/"
df_train, df_val, df_test = load_split_data_alexa(metadata_path, frac=frac, split1=split1, split2=split2)
print(f"Size of the training set: {len(df_train)}")
print(f"Size of the training set: {len(df_val)}")
print(f"Size of the training set: {len(df_test)}")

AudioContext.set(
    wavs_path="",
    frame_length=frame_length,
    frame_step=frame_step,
    fft_length=fft_length,
    add_suffix=False
)
lencoder = AudioContext.get().label_encoder
fencoder = AudioContext.get().feature_encoder

batch_size = 32
train_dataset = create_dataset(data_df=df_train, batch_size=batch_size)
validation_dataset = create_dataset(data_df=df_val, batch_size=batch_size)
test_dataset = create_dataset(data_df=df_test, batch_size=batch_size)

models = [
    ("model000",
     dict(),
     build_model_simple_logistic),
    ("model001",
     dict(rnn_layers=1, rnn_units=128, drop1=0.0, drop2=0.0),
     build_model_rnn),
    ("model002",
     dict(rnn_layers=2, rnn_units=128, drop1=0.0, drop2=0.0),
     build_model_rnn),
    ("model003",
     dict(rnn_layers=3, rnn_units=128, drop1=0.0, drop2=0.0),
     build_model_rnn),
    ("model004",
     dict(rnn_layers=4, rnn_units=128, drop1=0.0, drop2=0.0),
     build_model_rnn),
    ("model005",
     dict(rnn_layers=5, rnn_units=128, drop1=0.0, drop2=0.0),
     build_model_rnn),
]

# train the model
for model_config in models:
    train_model(model_config, save_path)

for model in models:
    model_name = model[0]
    print(f"Evaluating model {model_name} -----")

    # load a model
    model, model_res = load_model(model_name, save_path)

    # evaluate the model
    evaluate(model, test_dataset)

# TODO: analyze bias of the system, and think of ways to reduce them
# TODO: example: accents, regional speech difference, etc
# TODO: meeting saturday 12pm est, 9am pst
