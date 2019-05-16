#!python
# -*- coding: utf-8 -*-

import numpy as np
import random
import keras
import sys
from collections import Iterable
import pandas as pd
import imgaug as ia
import os


# from tqdm import tqdm ## This is unfortunately not installed on our deep machines.


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def img_data_reader(fname, required_shape=None, drange_out=None):
    """Read an image file with OpenCV
    The images are converted to float and rescaled to
    :param fname: File name of the image to read.
    :param required_shape: If not None, 'required_shape' must be a 2 or 3 tuple.
        None is returned if the image does not have the diemension requested.
    :param drange_out: If not None, then it must be a 2-tuple. The image is then converted to 'float' and
        its range converted to 'drange_out'. Example (0, 1) or (-1, 1).
    """
    import cv2
    img = cv2.imread(fname)
    if required_shape is not None and img.shape != required_shape:
        return None

    if drange_out is not None:
        img = adjust_dynamic_range(img.astype(np.float32), [0, 255], drange_out)

    return img


def read_data_list(list_fname, data_reader=None, max_len=0):
    """

    :param list_fname:
    :param data_reader: If not None, the files that are added to the list are only those for which this function does
        not return None. This is used to ceck that each file included to the list is valid.
    :param max_len: If higher than 0, return a list with at most that many elements.
    :return: list that contains the first word of each line.
    """
    files = []
    # with tqdm(desc='Reading file list', total=os.path.getsize(fname), unit='B', unit_scale=True, unit_divisor=1024,
    #           disable=not verbose) as pbar:
    with open(str(list_fname), 'rt') as fid:
        for line in fid:
            # pbar.update(len(line))
            words = line.split()
            fname = words[0]
            if data_reader is not None:
                if data_reader(fname) is not None:
                    files.append(fname)
            else:
                files.append(fname)
            if max_len > 0 and len(files) == max_len:
                break
    return files


class ListToSequence(keras.utils.Sequence):
    """Produce a Keras Sequence from from a file with a list of files. This sequence makes batches from a list of files.
    Inspired from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    At the end of each epoch, the list is shuffled.

    :param list_fname: List of file names.
    :param data_reader_fn: Function used to read a file. If this function returns None, the next file is used in the
        batch. Example: 'img_data_reader()'
    :param batch_size: Number of files in a batch.
    """

    def __init__(self, dl, reader_fn, batch_size=32, shuffle=True, kwargs=None):
        assert isinstance(dl, (Iterable, list, tuple))
        self.dl = dl
        self.reader_fn = reader_fn
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dl))
        self.shuffle = shuffle
        self.seeded = False
        self.kwargs = {} if kwargs is None else kwargs
        # assert len(self.dl) > self.batch_size, \
        #     ('The number of valid images in the list must be at least equal to the batch size: len(sequence)=%d, '
        #      'batch_size=%d') \
        #     % (len(self.dl), self.batch_size)
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.dl) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        if not self.seeded:
            seed = int.from_bytes(os.urandom(4), byteorder='little')
            ia.seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.seeded = True

        batch = []
        label = []
        for i in range(idx * self.batch_size, min(len(self.indexes), (idx + 1) * self.batch_size)):
            j = self.indexes[i]
            try:
                img, lbl = self.reader_fn(self.dl[j], **self.kwargs)
            except:
                print("Error on {}".format(self.dl[j]))
                raise
            batch.append(img)
            label.append(lbl)
        batch_inputs = reformat_batch_data(batch, np.stack)
        batch_labels = reformat_batch_data(label, np.stack)
        return batch_inputs, batch_labels


class SequenceOfSequences(keras.utils.Sequence):
    """Concatenate a list of sequences to produce a batch. This is helpful when the number of examples per category is
    not balanced. Then, creating one sequence per category and concatenating them with this class will equilibrate
    the number of examples per category.
    :param sequences: list of sequences
    """

    def __init__(self, sequences):
        assert isinstance(sequences, Iterable)
        self.sequences = sequences
        for seq in self.sequences:
            assert isinstance(seq, keras.utils.Sequence)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # print('SequenceOfSequences::on_epoch_end()')
        for seq in self.sequences:
            seq.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        min_len = sys.maxsize
        for seq in self.sequences:
            min_len = min(min_len, len(seq))
        return min_len

    def __getitem__(self, idx):
        """Generate one batch of data"""
        batch = None
        label = None
        for seq in self.sequences:
            b, l = seq[idx]
            if batch is None:
                batch, label = b, l
            else:
                batch = np.concatenate((batch, b))
                label = np.concatenate((label, l))
        return batch, label


class NoisySequence(keras.utils.Sequence):
    def __init__(self, shape=(16, 100)):
        self.shape = shape

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return 1

    def __getitem__(self, idx):
        return np.random.normal(0, 1, self.shape)


def reformat_batch_data(data_list, fusion_op=np.stack):
    """Intra or inter - batch concatenation
        Concatenate arrays while keeping the structure: list of arrays / dict of arrays / array

        Use fusion_op=np.stack for intra-batch (adds 1 dimension)
        Use fusion_op=np.concatenate for inter-batch (keeps dimensions)
        :param data_list: list of [list of arrays / dict of arrays / array]
        """
    assert (isinstance(data_list, list) and len(data_list) > 0)
    if isinstance(data_list[0], dict):
        d = dict()
        for v in data_list[0]:
            d[v] = fusion_op([dd[v] for dd in data_list])
    elif isinstance(data_list[0], list):
        d = list()
        for i in range(len(data_list[0])):
            d.append(fusion_op([dd[i] for dd in data_list]))
    else:
        d = fusion_op(data_list)
    return d


def get_subbatch_data(data, ids):
    """Get a subset of network input data (ie, sample on first axis)
        Get sub-arrays while keeping the structure: list of arrays / dict of arrays / array.
        Can be used on the output of get_one_epoch() for visualization of a subset of data

        :param data: list of arrays / dict of arrays / array
        :param ids: sample indexes in data
        """
    if isinstance(data, dict):
        d = dict()
        for v in data:
            d[v] = data[v][ids]
    elif isinstance(data, list):
        d = list()
        for i in range(len(data)):
            d.append(data[i][ids])
    else:
        d = data[ids]
    return d


class SequenceFromPandas(keras.utils.Sequence):
    """Produce a Keras Sequence from a pandas DataFrame
    Will produce 1 element for 1 DataFrame line.
    If len(df) is not a multiple of batch_size, will produce one more batch per epoch.
    :param df: pandas dataframe
    :param data_reader_fn: Function used to process one DataFrame line (formatted as a dict).
    :param batch_size: Number of elements in a batch.
    :param shuffle: whether to shuffle data between each epoch
    :param kwargs: additional args passed to data_reader_fn
    """

    def __init__(self, df, reader_fn, batch_size=32, shuffle=True, df2=None, add_nb=0, kwargs=None):
        assert isinstance(df, pd.DataFrame)
        self.df = df
        self.df2 = df2
        self.reader_fn = reader_fn
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.df))
        self.shuffle = shuffle
        self.seeded = False
        self.kwargs = {} if kwargs is None else kwargs

        self.indexes_df2 = np.arange(len(self.df2)) if add_nb > 0 else None
        self.add_nb = add_nb
        # assert self.df.shape[0] > self.batch_size, \
        #     ('The number of valid images in the list must be at least equal to the batch size: len(sequence)=%d, '
        #      'batch_size=%d') \
        #     % (self.df.shape[0], self.batch_size)
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.df.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        if not self.seeded:
            seed = int.from_bytes(os.urandom(4), byteorder='little')
            ia.seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.seeded = True

        batch = []
        label = []
        for i in range(idx * self.batch_size, min(len(self.indexes), (idx + 1) * self.batch_size)):
            j = self.indexes[i]
            # try:
            if self.add_nb:
                idx = np.random.choice(self.indexes_df2, self.add_nb)
                imgs2 = [ dict(self.df2.iloc[x]) for x in idx]
                img, lbl = self.reader_fn(dict(self.df.iloc[j]), imgs2, **self.kwargs)
            else:
                img, lbl = self.reader_fn(dict(self.df.iloc[j]), **self.kwargs)
            # except:
            #     print("Error on {}".format(self.df.iloc[j]))
            #     raise
            batch.extend(img)
            label.extend(lbl)
        batch_inputs = reformat_batch_data(batch, np.stack)
        batch_labels = reformat_batch_data(label, np.stack)
        return batch_inputs, batch_labels


def get_one_epoch(sequence):
    """Generate one epoch of a keras.utils.Sequence
            """
    ims = []
    lbls = []
    for i in range(len(sequence)):
        imb, lblb = sequence.__getitem__(i)
        ims.append(imb)
        lbls.append(lblb)
    return reformat_batch_data(ims, np.concatenate), reformat_batch_data(lbls, np.concatenate)

