import os
from os.path import join

import librosa
import numpy as np
import pyworld as pw
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import pdb
import pickle

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dir_to_wav', './dataset/vcc2018/wav', 'Dir to *.wav')
tf.app.flags.DEFINE_string('dir_to_bin', './dataset/vcc2018/bin', 'Dir to output *.bin')
tf.app.flags.DEFINE_integer('fs', 16000, 'Global sampling frequency')
tf.app.flags.DEFINE_float('f0_ceil', 500, 'Global f0 ceiling')

EPSILON = 1e-10
SETS = ['Training Set']
SPEAKERS = [s.strip() for s in tf.gfile.GFile('./etc/speakers.tsv', 'r').readlines()]
FFT_SIZE = 1024
SP_DIM = FFT_SIZE // 2 + 1
FEAT_DIM = SP_DIM + SP_DIM + 1 + 1 + 1  # [sp, ap, f0, en, s]
RECORD_BYTES = FEAT_DIM * 4  # all features saved in `float32`
EMB_DIM = 300
EMB_BYTES = EMB_DIM * 4


def wav2pw(x, fs=16000, fft_size=FFT_SIZE):
    """ Extract WORLD feature from waveform """
    _f0, t = pw.dio(x, fs, f0_ceil=args.f0_ceil)  # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size)  # extract aperiodicity
    return {
        'f0': f0,
        'sp': sp,
        'ap': ap,
    }


def extract(filename, fft_size=FFT_SIZE, dtype=np.float32):
    """ Basic (WORLD) feature extraction """
    x, _ = librosa.load(filename, sr=args.fs, mono=True, dtype=np.float64)
    features = wav2pw(x, args.fs, fft_size=fft_size)
    ap = features['ap']
    f0 = features['f0'].reshape([-1, 1])
    sp = features['sp']
    en = np.sum(sp + EPSILON, axis=1, keepdims=True)
    sp = np.log10(sp / en)
    return np.concatenate([sp, ap, f0, en], axis=1).astype(dtype)


def extract_and_save_bin_to(dir_to_bin, dir_to_source, sent_vec_dict):
    sets = [s for s in os.listdir(dir_to_source) if s in SETS]
    # print(sets)
    # pdb.set_trace()
    # sets is just training set
    for d in sets:
        path = join(dir_to_source, d)
        speakers = [s for s in os.listdir(path) if s in SPEAKERS]
        for s in speakers:
            path = join(dir_to_source, d, s)
            output_dir = join(dir_to_bin, d, s)
            if not tf.gfile.Exists(output_dir):
                tf.gfile.MakeDirs(output_dir)
            for f in os.listdir(path):
                filename = join(path, f)
                print(filename)
                if not os.path.isdir(filename):
                    features = extract(filename)
                    labels = SPEAKERS.index(s) * np.ones(
                        [features.shape[0], 1],
                        np.float32,
                    )
                    b = os.path.splitext(f)[0]
                    text_emb = sent_vec_dict[f]
                    # pdb.set_trace()
                    features = np.concatenate([features, labels], 1)
                    with open(join(output_dir, '{}.bin'.format(b)), 'wb') as fp:
                        fp.write(features.tostring())
                    with open(join(output_dir, 't_{}.bin'.format(b)), 'wb') as tp:
                        tp.write(text_emb.tostring())


class Tanhize(object):
    """ Normalizing `x` to [-1, 1] """

    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.xscale = xmax - xmin

    def forward_process(self, x):
        x = (x - self.xmin) / self.xscale
        return tf.clip_by_value(x, 0., 1.) * 2. - 1.

    def backward_process(self, x):
        return (x * .5 + .5) * self.xscale + self.xmin


def read(
        file_pattern,
        batch_size,
        record_bytes=RECORD_BYTES,
        capacity=256,
        min_after_dequeue=128,
        num_threads=8,
        format='NCHW',
        normalizer=None,
):
    """
    Read only `sp` and `speaker`

    Read only `sp` and `speaker`

    Return:
        `feature`: [b, c]
        `speaker`: [b,]
    """
    with tf.name_scope('InputSpectralFrame'):
        files = tf.gfile.Glob(file_pattern)
        filename_queue = tf.train.string_input_producer(files)

        reader = tf.FixedLengthRecordReader(record_bytes)
        _, value = reader.read(filename_queue)
        value = tf.decode_raw(value, tf.float32)

        value = tf.reshape(value, [FEAT_DIM, ])
        feature = value[:SP_DIM]  # NCHW format

        if normalizer is not None:
            feature = normalizer.forward_process(feature)

        if format == 'NCHW':
            feature = tf.reshape(feature, [1, SP_DIM, 1])
        elif format == 'NHWC':
            feature = tf.reshape(feature, [SP_DIM, 1, 1])
        else:
            pass
        speaker = tf.cast(value[-1], tf.int64)
        return tf.train.shuffle_batch(
            [feature, speaker],
            batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            num_threads=num_threads,
            # enqueue_many=True,
        )


def read_all(
        file_pattern,
        file_pattern2,
        batch_size,
        record_bytes=RECORD_BYTES,
        capacity=256,
        min_after_dequeue=128,
        num_threads=8,
        format='NCHW',
        normalizer=None,
):
    '''
    Read only `sp` and `speaker`
    Return:
        `feature`: [b, c]
        `speaker`: [b,]
    '''
    with tf.name_scope('InputSpectralFrame'):
        files = tf.gfile.Glob(file_pattern)
        filename_queue = tf.train.string_input_producer(files)

        reader = tf.FixedLengthRecordReader(record_bytes)
        _, value = reader.read(filename_queue)
        value = tf.decode_raw(value, tf.float32)

        value = tf.reshape(value, [FEAT_DIM, ])
        feature = value[:SP_DIM]  # NCHW format

        files2 = tf.gfile.Glob(file_pattern2)
        filename_queue2 = tf.train.string_input_producer(files2)
        reader2 = tf.FixedLengthRecordReader(EMB_BYTES)
        _, value2 = reader2.read(filename_queue2)
        value2 = tf.decode_raw(value2, tf.float32)

        if normalizer is not None:
            feature = normalizer.forward_process(feature)

        if format == 'NCHW':
            feature = tf.reshape(feature, [1, SP_DIM, 1])
            # text_emb = tf.reshape(value2, [1, 300, 1])
        elif format == 'NHWC':
            feature = tf.reshape(feature, [SP_DIM, 1, 1])
            # text_emb = tf.reshape(value2, [300, 1, 1])
        else:
            pass
        speaker = tf.cast(value[-1], tf.int64)

        # print(value2.shape)
        # tf_debug.
        text_emb = tf.reshape(value2, [EMB_DIM, ])
        # changed_file_pattern = file_pattern.split('/')
        # text_emb = tf.random_uniform(shape=(300,))
        # print(value)
        # pdb.set_trace()
        # pdb.set_trace()
        # tf_debug.LocalCLIDebugWrapperSession(sess)
        return tf.train.shuffle_batch(
            [feature, speaker, text_emb],
            batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            num_threads=num_threads,
            # enqueue_many=True,
        )


def read_whole_features(file_pattern, num_epochs=1):
    """
    Return
        `feature`: `dict` whose keys are `sp`, `ap`, `f0`, `en`, `speaker`
    """
    files = tf.gfile.Glob(file_pattern)
    print('{} files found'.format(len(files)))
    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    print("Processing {}".format(key), flush=True)
    value = tf.decode_raw(value, tf.float32)
    value = tf.reshape(value, [-1, FEAT_DIM])
    return {
        'sp': value[:, :SP_DIM],
        'ap': value[:, SP_DIM: 2 * SP_DIM],
        'f0': value[:, SP_DIM * 2],
        'en': value[:, SP_DIM * 2 + 1],
        'speaker': tf.cast(value[:, SP_DIM * 2 + 2], tf.int64),
        'filename': key,
    }


def pw2wav(features, feat_dim=513, fs=16000):
    ''' NOTE: Use `order='C'` to ensure Cython compatibility '''
    en = np.reshape(features['en'], [-1, 1])
    sp = np.power(10., features['sp'])
    sp = en * sp
    if isinstance(features, dict):
        return pw.synthesize(
            features['f0'].astype(np.float64).copy(order='C'),
            sp.astype(np.float64).copy(order='C'),
            features['ap'].astype(np.float64).copy(order='C'),
            fs,
        )
    features = features.astype(np.float64)
    sp = features[:, :feat_dim]
    ap = features[:, feat_dim:feat_dim * 2]
    f0 = features[:, feat_dim * 2]
    en = features[:, feat_dim * 2 + 1]
    en = np.reshape(en, [-1, 1])
    sp = np.power(10., sp)
    sp = en * sp
    return pw.synthesize(
        f0.copy(order='C'),
        sp.copy(order='C'),
        ap.copy(order='C'),
        fs
    )


if __name__ == '__main__':
    with open('./data/sent_emb.pkl', 'rb') as f:
        sent_vec_dict = pickle.load(f)
    extract_and_save_bin_to(
        args.dir_to_bin,
        args.dir_to_wav,
        sent_vec_dict,
    )
