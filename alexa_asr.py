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
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def download_file(url, name, save_path):
    """
    Helper function to download audio files and save them to a specified path
    with target name.

    Params:
    -------
    url (str): url for data
    save_path (str): local path for data to be saved
    name (str): local filename to append to save_path

    Returns:
    --------
    No return (refactoring opportunity to return an error if unsuccessful)
    """

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
    """
    Helper function to extract downloaded audio files in tar format.

    Params:
    -------
    file_path (str): local path for datafile
    mode (str): passes mode for tarfile.open function see tarfile docs for detail

    Returns:
    --------
    No return (refactoring opportunity to return an error if unsuccessful)
    """
    tar = tarfile.open(file_path, mode=mode)
    tar.extractall()
    tar.close()



def read_metadata(metadata_path, frac=1):
    """
    Function to read metadata (labels/transcription of audio files in our case)
    and select a random portion of data to be used.

    Params:
    -------
    metadata_path (str): path to directory with metadata
    frac (int): the randomly selected fraction of data you want to use. 1 means
                100%, or all data

    Returns:
    --------
    metadata_df (pd.dataframe): Data frame containing filename and transcription
    information for loaded files.
    """
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]]
    metadata_df = metadata_df.sample(frac=frac).reset_index(drop=True)

    return metadata_df


def load_split_data(metadata_path, wav_path, frac=1, split=0.9):
    """
    Function to load and split metadata for training and validation data sets.

    Params:
    -------
    metadata_path (str): path to directory with metadata
    frac (float): the randomly selectedfraction of data you want to use. 1 means 100%,
                  or all data.
    split (float): the fraction of data you want to use for training. 0.9 means 90%,
    rest is for test

    Returns:
    --------
    df_train (pd.dataframe): training set
    df_val (pd.dataframe): validation set
    """
    metadata_df = read_metadata(metadata_path, frac=frac)

    def to_fullpath(path):
        path_name = path + ".wav"
        full_path_list = [wav_path, path_name]
        return os.path.join(*full_path_list)

    metadata_df["file_name"] = metadata_df["file_name"].apply(to_fullpath)

    split = int(len(metadata_df) * split)
    df_train = metadata_df[:split]
    df_val = metadata_df[split:]

    return df_train, df_val


def load_split_data_alexa(metadata_path, frac=1, split1=0.8, split2=0.5):
    """
    Function to load and split alexa dataset into train/validate/test datasets

    Params:
    -------
    metadata_path (str): path to directory with metadata
    frac (float): the randomly selectedfraction of data you want to use. 1 means 100%,
                  or all data.
    split1 (float): the fraction of data you want to use for training.
                    .9 means 90%, rest is for validation and testing
    split2 (float): the fraction of data left after split1 to use for testing, rest
                    is for validation.

    Returns:
    --------
    df_train (pd.dataframe): training set
    df_val (pd.dataframe): validation set
    df_test (pd.dataframe): test set
    """

    all_files = []
    for subdir, dirs, files in os.walk(metadata_path):
        for file in files:
            full_path = os.path.join(subdir, file)
            all_files.append(full_path)

    df = pd.DataFrame(data={"file_name": all_files})
    df["normalized_transcription"] = "alexa"
    metadata_df = df
    # .sample actually randomizes the dataset
    metadata_df = metadata_df.sample(frac=frac).reset_index(drop=True)

    split1 = int(len(metadata_df) * split1)
    df_train = metadata_df[:split1]
    df_val = metadata_df[split1:]

    split2 = int(len(df_val) * split2)
    df_test = df_val[split2:]
    df_val = df_val[:split2]

    return df_train, df_val, df_test


class LabelEncoder:
    """
    The LabelEncoder class encodes labels for training and test data for ASR.

    Methods:
    -------
    __init__(self): class constructor
    encode(self, label):
    decode(self, nums):
    """
    def __init__(self):
        """
        Initialize LabelEncoder objects with all values set to initial state.
        Define tokens for encoding, define out of vocabulary token.
        """
        characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
        # encoder
        self.char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
        # decoder
        self.num_to_char = keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True
        )

    def encode(self, label):
        """
        Convert text label to tokens.
        """
        label = tf.strings.lower(label)
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        return self.char_to_num(label)

    def decode(self, nums):
        """
        Convert tokens back to text label.
        """
        return tf.strings.reduce_join(self.num_to_char(nums))


def wav_to_audio(filepath):
    """
    Function to decode WAV file to a 32 bit floating point tensor (waveform)
    with values from -1.0 to 1.0 with no dimensions of size 1.

    Params:
    -------
    filepath (str): path to directory with WAV files

    Returns:
    --------
    audio (tf.tensor): A waveform tensor of type float32 with values in the range (-1.0, 1.0)
    """
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
    """
    Function to convert audio waveform into spectrogram via a Short Time
    Fourier Transform (STFT).

    Params:
    -------
    audio (tf.tensor): A waveform tensor of type float32 with values in
    the range (-1.0, 1.0)
    frame_length (tf.scalar): An integer scalar tensor. The window length in samples
    frame_step (tf.scalar): An integer scalar tensor. The number of samples to step
    fft_length (tf.scalar): An integer scalar Tensor. The size of the fast fourier
    transform to apply. If not provided, uses the smallest power of 2 enclosing
    frame_length.

    Returns:
    --------
    spectrogram (tf.tensor): a tensor of STFT values representing a spectrogram
    (time frequency domain image).
    """
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
    """
    Function to convert a WAV audio file into a spectrogram to be used in machine
    learning models (thus 'wav to features'.

    Params:
    -------
    wav_file (str): WAV datafile name
    wavs_path (str): : local path for datafile
    frame_length (tf.scalar): An integer scalar tensor. The window length in samples
    frame_step (tf.scalar): An integer scalar tensor. The number of samples to step
    fft_length (tf.scalar): An integer scalar Tensor. The size of the fast fourier
    transform to apply. If not provided, uses the smallest power of 2 enclosing
    frame_length.
    add_suffix (bool):


    Returns:
    --------
    spectrogram (tf.tensor): a tensor of STFT values representing a spectrogram
    (time frequency domain image).
    """

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
    """
    The FeatureEncoder class encodes WAV files as normalized spectrograms to be used
    as features for models performing ASR.

    Methods:
    -------
    __init__(self): class constructor
    features(self, wav_file):
    """
    def __init__(
            self,
            wavs_path,
            frame_length,
            frame_step,
            fft_length,
            add_suffix,
    ):
        """
        Initialize FeatureEncoder objects with all values set to initial state.
        """
        self.wavs_path = wavs_path
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.add_suffix = add_suffix

    def features(
            self,
            wav_file,
    ):
        """
        Convert WAV files to normalized spectrograms.
        """
        return wav_to_features(
            wav_file=wav_file,
            wavs_path=self.wavs_path,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            add_suffix=self.add_suffix
        )


class AudioContext:
    """
    The AudioContext class is a helper class to encapsulate methods
    to encode WAV files and corresponding lables as well as to decode
    predictions to english text.

    Methods:
    -------
    __init__(self,
             wavs_path,
             frame_length,
             frame_step,
             fft_length,
             add_suffix): class constructor
    set(self,
        wavs_path,
        frame_length,
        frame_step,
        fft_length,
        add_suffix): static setter method
    get(): static getter method
    """
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
        """
        Check ability to instatiate AudioContext, create LabelEncoder, FeatureEncoder
        objects.
        """
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
        """
        Create AudioContext object. Block future creation of AudioContext for
        safety.
        """
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
        """
        Retrieve AudioContext object instance.
        """
        if AudioContext._instance is None:
            raise ValueError("AudioContext not set yet")

        return AudioContext._instance


def encode_single_sample(wav_file, label):
    """
    Function to both encode label and features for a single WAV file.

    Params:
    -------
    wav_file (str): WAV datafile name
    label (str): WAV file transcript


    Returns:
    --------
    features (tf.tensor): a tensor of STFT values representing a spectrogram
    (time frequency domain image)
    encoded_label (int): an integer which represents the WAV transcript as an integer
    where every digit represents the index within the vocabulary defined in class
    LabelEncoder.
    """
    lencoder = AudioContext.get().label_encoder
    fencoder = AudioContext.get().feature_encoder

    features = fencoder.features(wav_file)
    encoded_label = lencoder.encode(label)

    return features, encoded_label


def create_dataset(data_df, batch_size):
    """
    Function to generate a dataset for use in model training, validation or
    testing. Dataset creation is optimized using tf.data.

    Params:
    -------
    data_df (pd.dataframe): Metadata dataframe.
    batch_size(int): length to pad any data less than the given constant


    Returns:
    --------
    features (tf.tensor): a tensor of STFT values representing a spectrogram
    (time frequency domain image)
    encoded_label (int): an integer which represents the WAV transcript as an integer
    where every digit represents the index within the vocabulary defined in class
    LabelEncoder.
    """
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
    """
    Function to compute the training time loss value using the
    Connectionist Temporal Classification Loss, or CTC Loss algorithm.

    Params:
    -------
    y_true (): Target label.
    y_pred(): Predicted label.


    Returns:
    --------
    loss (tf.tensor): tensor with shape (samples,1) containing the CTC loss
    of each element.
    """
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def decode_batch_predictions(pred):
    """
    Function to decode a batch of predictions from the model.
    Since we are using CTC loss, uses ctc.decode to translate
    output of softmax, then map to our vocabulary using the label
    decoder class.
    Useful code borrowed from: https://keras.io/examples/audio/ctc_asr/

    Params:
    -------
    pred (tf.tensor): tensor containing prediction outputs.


    Returns:
    --------
    res (str): Predicted text from model.
    """
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
    """
    Function to decode a single prediction from model.

    Params:
    -------
    pred (tf.tensor): tensor containing prediction output.


    Returns:
    --------
    res (str): Predicted text from model.
    """
    pred = tf.reshape(pred, [1, pred.shape[0], pred.shape[1]])
    return decode_batch_predictions(pred)


def accuracy(labels, predictions):
    """
    A function to calculate the relative accuracy of a prediction
    to the target label.

    Params:
    -------
    labels (tf.tensor): tensor containing prediction output.


    Returns:
    --------
    res (str): Predicted text from model.
    """
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
    """
    A function to calculate the absolute accuracy of a prediction
    to the target label.

    Params:
    -------
    labels (tf.tensor): tensor containing prediction output.


    Returns:
    --------
    res (str): Predicted text from model.
    """
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
    """
    The ModelEvaluator Class to calculates and displays a batch
    of outputs after each training epoch to display relevant
    statistics and predicted vs target labels.


    Methods:
    -------
    __init__(self,
             dataset,
              model): class constructor

    print (wer_score,
           res_acc,
           abs_acc,
           pred_df): static method to override print()

    do_prediction (self): method to execute model.predict on
                          data set, decode predictions, calculate
                          relative accuracy, calculate absolute accuracy,
                          and calculate word error rate.

    on_epoch_end (self,
                  epoch,
                  logs): method to execute at the end of each epoch to
                         make predictions and print predictions,
                         relative accuracy, absolute accuracy,
                         and word error rate.

    """
    def __init__(self, dataset, model):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.history = []

    @staticmethod
    def print(wer_score, res_acc, abs_acc, pred_df):
        """
        Override default print() for our context.
        """
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        print(f"Accuracy: {res_acc:.4f}")
        print("-" * 100)
        print(f"Abs accuracy: {abs_acc:.4f}")
        print("-" * 100)
        print(pred_df.sample(n=5))

    def do_prediction(self):
        """
        Execute model.predict() and calculate relevant statistics.
        """
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
        """
        Execute model.predict() and calculate relevant statistics. Add output to
        history. Print model prediction and statistics to console.
        """
        wer_score, res_acc, abs_acc, pred_df = self.do_prediction()
        self.history.append((wer_score, res_acc, abs_acc, pred_df))
        self.print(wer_score, res_acc, abs_acc, pred_df)


def evaluate(model, test_dataset):
    """
    Function to evaluate trained model against test dataset.

    Params:
    -------
    model (keras.model: Trained model.
    test_dataset (): Test dataset


    Returns:
    --------
    wer_score ():
    res_acc ():
    abs_acc ():
    pred_df ():
    """
    print("evaluate model against test dataset")
    test_evaluator = ModelEvaluator(
        dataset=test_dataset,
        model=model
    )
    wer_score, res_acc, abs_acc, pred_df = test_evaluator.do_prediction()
    test_evaluator.print(wer_score, res_acc, abs_acc, pred_df)
    return wer_score, res_acc, abs_acc, pred_df


def write_pickle(data, file, ensure_exist=True):
    """
    Function to save data into specified pickle file

    Params:
    -------
    data: data to save
    file: file path to save data to
    ensure_exist: create file path if does not exist


    Returns:
    --------
    None
    """
    if ensure_exist:
        path = os.path.dirname(file)
        if len(path) > 0:
            os.makedirs(path, exist_ok=True)

    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(file):
    """
    Function to read pickle file

    Params:
    -------
    file: file path to read data from

    Returns:
    --------
    data loaded from pickle file
    """
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
        learning_rate=1e-3,
):
    """
    reference: https://keras.io/examples/audio/ctc_asr/
    """
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
    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)

    return model


def build_model_simple_logistic(
        input_dim,
        output_dim,
        model_name,
        learning_rate=1e-3,
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
    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)

    return model


def build_model_ann(
        input_dim,
        output_dim,
        model_name,
        hidden_layers=1,
        hidden_units=[256],
        learning_rate=1e-3,
):
    """build artificial nueral network model"""
    # Model's input
    input_spectrogram = layers.Input(
        (None, input_dim),
        name=f"{model_name}_input"
    )

    x = input_spectrogram
    for i in range(hidden_layers):
        # Dense layer
        x = layers.Dense(
            units=hidden_units[i],
            name=f"{model_name}_dense_{i}"
        )(x)
        x = layers.ReLU(name=f"{model_name}_dense_{i}_relu")(x)

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
    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)

    return model


def train_model(
        input_dim,
        epochs,
        model_config,
        save_path,
        train_dataset,
        validation_dataset,
        check_exist=True
):
    """
    Train model with train/validation dataset, save results into pickle
    """
    model_name = model_config[0]
    if check_exist and exists(os.path.join(save_path, model_name)):
        print(f"model {model_name} already exists")
        return

    model_kwargs = model_config[1]
    model_builder = model_config[2]

    model = model_builder(
        input_dim=input_dim,
        output_dim=AudioContext.get().label_encoder.char_to_num.vocabulary_size(),
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
    """
    Load trained model from pickle file
    """
    model = keras.models.load_model(
        os.path.join(save_path, model_name),
        custom_objects={"CTCLoss": CTCLoss})
    model_res = read_pickle(os.path.join(save_path, f"{model_name}_validation.pkl"))

    return model, model_res


def encode_sample(idx, df):
    """
    Encode a label at the index, idx, to a list of numbers
    """
    features, label = encode_single_sample(
        df.iloc[idx].file_name,
        df.iloc[idx].normalized_transcription
    )

    return features, label


def kmean_cluster(df_train, num_clusters=32, save_path="C:\data\kmean"):
    """
    do kmean cluster on a dataset and save the clustered kmean model to database

    Params:
    -------
    df_train: dataset to do kmean clustering on
    num_clusters: number of clusters to create
    save_path: path to save results to in pickle file format

    Returns:
    --------
    data_dict: a data dictionary of clustered features
    km: trained cluster model
    df_train: input dataset with clusters assigned
    """
    samples_path = os.path.join(save_path, "kmean_samples.pkl")
    if not os.path.exists(samples_path):
        all_features = None
        for idx in range(len(df_train)):
            features, label = encode_sample(idx, df_train)
            all_features = \
                tf.concat([all_features, features], axis=0) \
                    if all_features is not None else features

        all_samples = np.array(all_features)
        write_pickle(all_samples, samples_path)
    else:
        print(f"loading {samples_path}")
        all_samples = read_pickle(samples_path)

    km_path = os.path.join(save_path, "km.pkl")
    if not os.path.exists(km_path):
        km = KMeans(n_clusters=num_clusters)
        km.fit(all_samples)
        write_pickle(km, km_path)
    else:
        print(f"loading {km_path}")
        km = read_pickle(km_path)

    features_df_path = os.path.join(save_path, "kmean_clustered_data.pkl")
    if not os.path.exists(features_df_path):
        clusters = km.labels_.tolist()
        cfeatures_list = []
        count_idx = 0
        for idx in range(len(df_train)):
            features, label = encode_sample(idx, df_train)
            cfeatures = clusters[count_idx:count_idx + len(features)]
            cfeatures_list.append(cfeatures)
            count_idx += len(features)

        df_train["cfeatures"] = cfeatures_list

        lencoder = AudioContext.get().label_encoder
        df_train["label"] = df_train["normalized_transcription"].apply(lencoder.encode)

        write_pickle(df_train, features_df_path)
    else:
        print(f"loading {features_df_path}")
        df_train = read_pickle(features_df_path)

    clusted_dict_path = os.path.join(save_path, "kmean_clustered_dict.pkl")
    if not os.path.exists(clusted_dict_path):
        data_dict = dict(zip(df_train.file_name, df_train.cfeatures))
        write_pickle(data_dict, clusted_dict_path)
    else:
        print(f"loading {clusted_dict_path}")
        data_dict = read_pickle(clusted_dict_path)


    return data_dict, km, df_train


def kmean_convert(df_test, save_path="C:\data\kmean"):
    """assign clusters to input dataset with specified kmean model"""
    all_features = None
    for idx in range(len(df_test)):
        features, label = encode_sample(idx, df_test)
        all_features = \
            tf.concat([all_features, features], axis=0) \
                if all_features is not None else features

    all_samples = np.array(all_features)

    km_path = os.path.join(save_path, "km.pkl")
    km = read_pickle(km_path)

    def get_closest_center(ft, centers):
        cur_dot = None
        cur_center = None
        for idx, center in enumerate(centers):
            new_dot = np.dot(ft, center)
            if cur_dot is None or new_dot < cur_dot:
                cur_dot = new_dot
                cur_center = idx

        return cur_center

    # use the clusters in km to assign clusters to each feature
    centers = km.cluster_centers_
    clusters = []
    for ft in all_features:
        cluster = get_closest_center(ft, centers)
        clusters.append(cluster)

    cfeatures_list = []
    count_idx = 0
    for idx in range(len(df_test)):
        features, label = encode_sample(idx, df_test)
        cfeatures = clusters[count_idx:count_idx + len(features)]
        cfeatures_list.append(cfeatures)
        count_idx += len(features)

    df_test["cfeatures"] = cfeatures_list

    lencoder = AudioContext.get().label_encoder
    df_test["label"] = df_test["normalized_transcription"].apply(lencoder.encode)

    return df_test


def data_generator(data_df):
    """data generator to feed data input keras"""
    train_size = len(data_df)
    idx = 0
    while True:
        x_train = tf.reshape(tf.convert_to_tensor(data_df.loc[idx]["cfeatures"]), [1, -1, 1])
        y_train = tf.reshape(data_df.loc[idx]["label"], [1, -1])
        idx += 1
        idx = idx % train_size
        yield x_train, y_train


class DGenModelEvaluator(keras.callbacks.Callback):
    """model evaluator for data generator datasets"""
    def __init__(self, ds, model):
        super().__init__()

        ds_dgen = data_generator(ds)
        ds_size = len(ds)

        self.dgen = ds_dgen
        self.size = ds_size
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
        for _ in range(self.size):
            X, y = next(self.dgen)
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


def evaluate_km(model, test_dataset):
    """evaluate model using data generator evaluator"""
    print("evaluate model against test dataset")
    test_evaluator = DGenModelEvaluator(
        ds=test_dataset,
        model=model
    )
    wer_score, res_acc, abs_acc, pred_df = test_evaluator.do_prediction()
    test_evaluator.print(wer_score, res_acc, abs_acc, pred_df)
    return wer_score, res_acc, abs_acc, pred_df


def train_km_model(
        epochs,
        model_config,
        save_path,
        train_dataset,
        validation_dataset,
        check_exist=True
):
    """train model on kmean clustered data"""
    train_dgen = data_generator(train_dataset)
    train_size = len(train_dataset)

    model_name = model_config[0]
    if check_exist and exists(os.path.join(save_path, model_name)):
        print(f"model {model_name} already exists")
        return

    model_kwargs = model_config[1]
    model_builder = model_config[2]

    model = model_builder(
        input_dim=1,
        output_dim=AudioContext.get().label_encoder.char_to_num.vocabulary_size(),
        model_name=model_name,
        **model_kwargs,
    )
    model.summary(line_length=110)

    # create validation evaluator
    validation_callback = DGenModelEvaluator(
        ds=validation_dataset,
        model=model
    )

    # Train the model
    history = model.fit_generator(
        train_dgen,
        steps_per_epoch=train_size,
        epochs=epochs,
        callbacks=[validation_callback],
    )

    # save the model
    model.save(os.path.join(save_path, model_name))
    write_pickle(
        {
            # "validation results": validation_callback.history,
            "train history": history.history
        },
        os.path.join(save_path, f"{model_name}_validation.pkl")
    )


def train_alexa():
    # params
    frame_length = 256
    frame_step = 160
    fft_length = 384
    epochs = 10
    frac = 1
    split1 = 0.8  # using 80% of the input data to train, 20% for validation
    split2 = 0.5  # using 50% of validation for model tuning, 50% for testing
    lj_frac = 1
    lj_split = 0.9
    save_path = "C:/data"

    metadata_path = "C:/Users/aphri/pycharm/t0001/data/alexa1/"
    df_train, df_val, df_test = load_split_data_alexa(metadata_path, frac=frac, split1=split1, split2=split2)
    print(f"Size of the training set: {len(df_train)}")
    print(f"Size of the val set: {len(df_val)}")
    print(f"Size of the test set: {len(df_test)}")

    metadata_path_lj = "C:/data/LJSpeech-1.1/metadata.csv"
    wav_path_lj = "C:/data/LJSpeech-1.1/wavs"
    df_train_lj, df_val_lj = load_split_data(metadata_path_lj, wav_path_lj, frac=lj_frac, split=lj_split)
    print(f"lj Size of the training set: {len(df_train_lj)}")
    print(f"lj Size of the val set: {len(df_val_lj)}")

    AudioContext.set(
        wavs_path="",
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        add_suffix=False
    )

    batch_size = 32
    train_dataset = create_dataset(data_df=df_train, batch_size=batch_size)
    validation_dataset = create_dataset(data_df=df_val, batch_size=batch_size)
    test_dataset = create_dataset(data_df=df_test, batch_size=batch_size)
    test_dataset_lj = create_dataset(data_df=df_val_lj, batch_size=batch_size)

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
        ("model006",
         dict(rnn_layers=5, rnn_units=64, drop1=0.0, drop2=0.0),
         build_model_rnn),
        ("model007",
         dict(rnn_layers=0, rnn_units=128, drop1=0.0, drop2=0.0),
         build_model_rnn),
        ("model001ann",
         dict(hidden_layers=1, hidden_units=[256]),
         build_model_ann),
        ("model002ann",
         dict(hidden_layers=2, hidden_units=[256, 128]),
         build_model_ann),
        ("model003ann",
         dict(hidden_layers=3, hidden_units=[256, 128, 64]),
         build_model_ann),
    ]

    # train the model
    for model_config in models:
        train_model(
            input_dim=fft_length // 2 + 1,
            model_config=model_config,
            save_path=save_path,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
        )

    model_list = []
    name_list = []
    rnn_layers_list = []
    rnn_units_list = []
    wer_list = []
    racc_list = []
    aacc_list = []
    wer_list_lj = []
    racc_list_lj = []
    aacc_list_lj = []

    name_map = {
        "model000": "Simple Logistic Multiclassification",
        "model001": "Recurrent Neural Network",
        "model002": "Recurrent Neural Network",
        "model003": "Recurrent Neural Network",
        "model004": "Recurrent Neural Network",
        "model005": "Recurrent Neural Network",
        "model006": "Recurrent Neural Network",
        "model007": "Recurrent Neural Network",
        "model001ann": "Artificial Neural Network",
        "model002ann": "Artificial Neural Network",
        "model003ann": "Artificial Neural Network",
    }
    for model in models:
        model_name = model[0]
        model_params = model[1]
        print(f"Evaluating model {model_name} -----")

        # load a model
        model, model_res = load_model(model_name, save_path)

        # evaluate the model
        wer_score, res_acc, abs_acc, pred_df = evaluate(model, test_dataset)

        model_list.append(model_name)
        name_list.append(name_map[model_name])
        rnn_layers_list.append(model_params["rnn_layers"] if "rnn_layers" in model_params else "-")
        rnn_units_list.append(model_params["rnn_units"] if "rnn_units" in model_params else "-")
        wer_list.append(wer_score)
        racc_list.append(res_acc)
        aacc_list.append(abs_acc)

        # evaluate the model
        wer_score_lj, res_acc_lj, abs_acc_lj, pred_df_lj = evaluate(model, test_dataset_lj)
        wer_list_lj.append(wer_score_lj)
        racc_list_lj.append(res_acc_lj)
        aacc_list_lj.append(abs_acc_lj)

    res = pd.DataFrame(
        data={
            "Model": model_list,
            "Name": name_list,
            "RNN Layers": rnn_layers_list,
            "RNN Units": rnn_units_list,
            "Alexa DS Relative Accuracy": racc_list,
            "LJ DS Relative Accuracy": racc_list_lj,
            "Alexa DS Absolute Accuracy": aacc_list,
            "LJ DS Absolute Accuracy": aacc_list_lj,
            "Alexa DS Word Error Rate": wer_list,
            "LJ DS Word Error Rate": wer_list_lj,
        }
    )
    res.to_csv(os.path.join(save_path, "alexa_res.csv"))


def train_lj():

    # params
    frame_length = 256
    frame_step = 160
    fft_length = 384
    epochs = 10
    frac = 1
    split1 = 0  # using 0% of the input data to train, 100% for validation
    split2 = 0.5  # using 50% of validation for model tuning, 50% for testing
    lj_frac = 1  # using 100% of lj speech data for training
    lj_split = 0.5  # use 50% data for training, 50% for validation
    save_path = "C:/data"

    metadata_path = "C:/Users/aphri/pycharm/t0001/data/alexa1/"
    df_train, df_val, df_test = load_split_data_alexa(metadata_path, frac=frac, split1=split1, split2=split2)
    print(f"Size of the training set: {len(df_train)}")
    print(f"Size of the val set: {len(df_val)}")
    print(f"Size of the test set: {len(df_test)}")

    metadata_path_lj = "C:/data/LJSpeech-1.1/metadata.csv"
    wav_path_lj = "C:/data/LJSpeech-1.1/wavs"
    df_train_lj, df_test_lj = load_split_data(metadata_path_lj, wav_path_lj, frac=lj_frac, split=lj_split)
    print(f"lj Size of the training set: {len(df_train_lj)}")
    print(f"lj Size of the test set: {len(df_test_lj)}")

    AudioContext.set(
        wavs_path="",
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        add_suffix=False
    )

    batch_size = 32
    train_dataset = create_dataset(data_df=df_train_lj, batch_size=batch_size)
    validation_dataset = create_dataset(data_df=df_val, batch_size=batch_size)
    test_dataset = create_dataset(data_df=df_test, batch_size=batch_size)
    test_dataset_lj = create_dataset(data_df=df_test_lj, batch_size=batch_size)

    models = [
        ("model005lj_half",
         dict(rnn_layers=5, rnn_units=128, drop1=0.0, drop2=0.0),
         build_model_rnn),
        ("model006lj_half",
         dict(rnn_layers=5, rnn_units=64, drop1=0.0, drop2=0.0),
         build_model_rnn),
    ]

    # train the model
    for model_config in models:
        train_model(
            input_dim=fft_length // 2 + 1,
            model_config=model_config,
            save_path=save_path,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
        )

    model_list = []
    name_list = []
    rnn_layers_list = []
    rnn_units_list = []
    wer_list = []
    racc_list = []
    aacc_list = []
    wer_list_lj = []
    racc_list_lj = []
    aacc_list_lj = []

    name_map = {
        "model005lj_half": "Recurrent Neural Network",
        "model006lj_half": "Recurrent Neural Network",
    }
    for model in models:
        model_name = model[0]
        model_params = model[1]
        print(f"Evaluating model {model_name} -----")

        # load a model
        model, model_res = load_model(model_name, save_path)

        # evaluate the model
        wer_score, res_acc, abs_acc, pred_df = evaluate(model, test_dataset)

        model_list.append(model_name)
        name_list.append(name_map[model_name])
        rnn_layers_list.append(model_params["rnn_layers"] if "rnn_layers" in model_params else "-")
        rnn_units_list.append(model_params["rnn_units"] if "rnn_units" in model_params else "-")
        wer_list.append(wer_score)
        racc_list.append(res_acc)
        aacc_list.append(abs_acc)

        # evaluate the model
        wer_score_lj, res_acc_lj, abs_acc_lj, pred_df_lj = evaluate(model, test_dataset_lj)
        wer_list_lj.append(wer_score_lj)
        racc_list_lj.append(res_acc_lj)
        aacc_list_lj.append(abs_acc_lj)

    res = pd.DataFrame(
        data={
            "Model": model_list,
            "Name": name_list,
            "RNN Layers": rnn_layers_list,
            "RNN Units": rnn_units_list,
            "Alexa DS Relative Accuracy": racc_list,
            "LJ DS Relative Accuracy": racc_list_lj,
            "Alexa DS Absolute Accuracy": aacc_list,
            "LJ DS Absolute Accuracy": aacc_list_lj,
            "Alexa DS Word Error Rate": wer_list,
            "LJ DS Word Error Rate": wer_list_lj,
        }
    )
    res.to_csv(os.path.join(save_path, "lj_res.csv"))


def train_km():
    # params
    frame_length = 256
    frame_step = 160
    fft_length = 384
    epochs = 10
    frac = 1
    split1 = 0.8 # using 80% of the input data to train
    split2 = 0.5 # using 50% for validation, 50% for testing
    lj_frac = 1
    lj_split = 0.9
    save_path = "C:/data"
    kmean_save_path = os.path.join(save_path, "kmean")

    metadata_path = "C:/Users/aphri/pycharm/t0001/data/alexa1/"
    df_train, df_val, df_test = load_split_data_alexa(
        metadata_path,
        frac=frac,
        split1=split1,
        split2=split2
    )
    print(f"Size of the training set: {len(df_train)}")
    print(f"Size of the val set: {len(df_val)}")
    print(f"Size of the test set: {len(df_test)}")

    metadata_path_lj = "C:/data/LJSpeech-1.1/metadata.csv"
    wav_path_lj = "C:/data/LJSpeech-1.1/wavs"
    df_train_lj, df_val_lj = load_split_data(metadata_path_lj, wav_path_lj, frac=lj_frac, split=lj_split)
    print(f"lj Size of the training set: {len(df_train_lj)}")
    print(f"lj Size of the val set: {len(df_val_lj)}")

    AudioContext.set(
        wavs_path="",
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        add_suffix=False
    )

    print("run kmean")
    data_dict, km, data_df_train = kmean_cluster(
        df_train=df_train,
        num_clusters=32,
        save_path=kmean_save_path,
    )
    data_df_train = data_df_train.reset_index(drop=True)

    print("convert kmean for alexa val")
    data_df_validation = kmean_convert(
        df_test=df_val,
        save_path=kmean_save_path
    )
    data_df_validation = data_df_validation.reset_index(drop=True)

    print("convert kmean for alexa test")
    data_df_test = kmean_convert(
        df_test=df_test,
        save_path=kmean_save_path
    )
    data_df_test = data_df_test.reset_index(drop=True)

    print("convert kmean for lj test")
    data_df_test_lj = kmean_convert(
        df_test=df_val_lj,
        save_path=kmean_save_path
    )
    data_df_test_lj = data_df_test_lj.reset_index(drop=True)

    models = [
        ("model000km",
         dict(),
         build_model_simple_logistic),
        ("model001km",
         dict(rnn_layers=1, rnn_units=128, drop1=0.0, drop2=0.0),
         build_model_rnn),
        ("model002km",
         dict(rnn_layers=2, rnn_units=128, drop1=0.0, drop2=0.0),
         build_model_rnn),
        ("model003km",
         dict(rnn_layers=3, rnn_units=128, drop1=0.0, drop2=0.0),
         build_model_rnn),
        ("model004km",
         dict(rnn_layers=4, rnn_units=128, drop1=0.0, drop2=0.0),
         build_model_rnn),
        ("model005km",
         dict(rnn_layers=5, rnn_units=128, drop1=0.0, drop2=0.0),
         build_model_rnn),
        ("model006km",
         dict(rnn_layers=5, rnn_units=64, drop1=0.0, drop2=0.0),
         build_model_rnn),
    ]

    print("training models")
    # train the model
    for model_config in models:
        print(f"training model: {model_config[0]}")
        train_km_model(
            epochs=epochs,
            model_config=model_config,
            save_path=save_path,
            train_dataset=data_df_train,
            validation_dataset=data_df_validation,
        )

    model_list = []
    name_list = []
    rnn_layers_list = []
    rnn_units_list = []
    wer_list = []
    racc_list = []
    aacc_list = []
    wer_list_lj = []
    racc_list_lj = []
    aacc_list_lj = []

    name_map = {
        "model000km": "KMean, Multi-class Classification",
        "model001km": "KMean, Recurrent Neural Network",
        "model002km": "KMean, Recurrent Neural Network",
        "model003km": "KMean, Recurrent Neural Network",
        "model004km": "KMean, Recurrent Neural Network",
        "model005km": "KMean, Recurrent Neural Network",
        "model006km": "KMean, Recurrent Neural Network",
    }
    for model in models:
        model_name = model[0]
        model_params = model[1]
        print(f"Evaluating model {model_name} -----")

        # load a model
        model, model_res = load_model(model_name, save_path)

        # evaluate the model
        wer_score, res_acc, abs_acc, pred_df = evaluate_km(model, data_df_test)

        model_list.append(model_name)
        name_list.append(name_map[model_name])
        rnn_layers_list.append(model_params["rnn_layers"] if "rnn_layers" in model_params else "-")
        rnn_units_list.append(model_params["rnn_units"] if "rnn_units" in model_params else "-")
        wer_list.append(wer_score)
        racc_list.append(res_acc)
        aacc_list.append(abs_acc)

        # evaluate the model
        wer_score_lj, res_acc_lj, abs_acc_lj, pred_df_lj = evaluate_km(model, data_df_test_lj)
        wer_list_lj.append(wer_score_lj)
        racc_list_lj.append(res_acc_lj)
        aacc_list_lj.append(abs_acc_lj)

    res = pd.DataFrame(
        data={
            "Model": model_list,
            "Name": name_list,
            "RNN Layers": rnn_layers_list,
            "RNN Units": rnn_units_list,
            "Alexa DS Relative Accuracy": racc_list,
            "LJ DS Relative Accuracy": racc_list_lj,
            "Alexa DS Absolute Accuracy": aacc_list,
            "LJ DS Absolute Accuracy": aacc_list_lj,
            "Alexa DS Word Error Rate": wer_list,
            "LJ DS Word Error Rate": wer_list_lj,
        }
    )

    # using Alexa dataset only, 80% training (295), 10% validation (37), 10% test (37)
    print(res)
    res.to_csv(os.path.join(save_path, "alexa_res_kmean.csv"))


if __name__ == "__main__":
    train_alexa()
    train_lj()
    train_km()
