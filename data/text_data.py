from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.data.functional import generate_sp_model
from torchtext.data.functional import load_sp_model, sentencepiece_numericalizer
import numpy as np
import sentencepiece as spm

TOKENS_FILE = "tokens.txt"

def chunk_text(texts, chunk_length, y_chunk_length):
    return [(t[:chunk_length], t[chunk_length:(chunk_length + y_chunk_length)]) for t in texts if (chunk_length + y_chunk_length) <= len(t)]

def load_data_list(data, key, subset):
    selection = data[key]
    if subset:
        selection = selection[subset]
    return selection

class TextDatasetWrapper:
    name = None
    version = None
    splits = ["train", "test", "validation"]
    split_lengths = None
    subset = None
    x_length = None
    target_length = None

    def __init__(self, vocab_size):
        self.sp_vocab_size = vocab_size
        self.vocab_size = vocab_size + 1
        # Add one for the stop token at the end of the sequence
        self.y_length = self.target_length + 1

        self.data = {}
        self.split_data = {}
        self.encoded_data = {}
        self.unk_token = 0
        self.start_token = 1
        self.stop_token = 2
        self.pad_token = vocab_size

        self.extract_data()
        tokenizer_data = ""
        for split in self.splits:
            x, target = self.split_x_target(self.data[split])
            self.split_data[split] = {"x": x, "target": target}
            tokenizer_data += "\n".join(x)
            tokenizer_data += "\n".join(target)
        self.train_tokenizer(tokenizer_data)

        self.model_file = f"{self.name}.model"
        self.sp_model = load_sp_model(self.model_file)
        self.encoding_generator = sentencepiece_numericalizer(self.sp_model)
        self.sp_base = spm.SentencePieceProcessor(self.model_file)

        for split in self.splits:
            x, target = self.encode_data(self.split_data[split]["x"], self.split_data[split]["target"])
            x, target = self.trim_length(x, target)
            if not self.x_length:
                self.x_length = max([len(s) for s in x])
            if not self.target_length:
                self.target_length = max([len(s) for s in target])
            x = self.pad_sequences(x, self.x_length)
            target = self.pad_sequences(target, self.y_length, use_stop=True)
            self.encoded_data[split] = {"x": x, "target": target}
        self.create_final_sets()

    def extract_data(self):
        dataset = load_dataset(self.name, self.version)
        for i, split in enumerate(self.splits):
            s_data = load_data_list(dataset, split, self.subset)
            if self.split_lengths and self.split_lengths[i]:
                s_data = s_data[:self.split_lengths[i]]
            self.data[split] = s_data

    def split_x_target(self, split):
        """Override.  Should return x and y"""
        pass

    def trim_length(self, x, target):
        """Optional override to reduce length."""
        return x, target

    def create_final_sets(self):
        """Optional override to split up encoded data sets.  Useful if the initial data doesn't have a validation set, for example."""
        self.final_data = self.encoded_data

    def train_tokenizer(self, tokenizer_data):
        with open(TOKENS_FILE, "w+") as f:
            f.write(tokenizer_data)

        generate_sp_model(TOKENS_FILE, vocab_size=self.sp_vocab_size, model_prefix=self.name)

    def encode_data(self, x, target):
        encoded_x = list(self.encoding_generator(x))
        encoded_target = list(self.encoding_generator(target))
        return encoded_x, encoded_target

    def decode_ids(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = list(ids.numpy())
        new_ids = []
        for i in ids:
            i = int(i)
            if i in [self.stop_token, self.pad_token]:
                break
            elif i in [self.start_token]:
                continue
            new_ids.append(i)

        return self.sp_base.decode(new_ids)

    def decode_batch(self, id_tensor):
        decoded = []
        for i in range(id_tensor.shape[0]):
            decoded.append(self.decode_ids(id_tensor[i, :]))
        return decoded

    def pad_sequences(self, seqs, length, use_stop=False):
        return [self.pad_sequence(s, length, use_stop) for s in seqs]

    def pad_sequence(self, seq, length, use_stop):
        if use_stop:
            seq = seq + [self.stop_token]
        if len(seq) < length:
            seq = seq + [self.pad_token] * (length - len(seq))
        return seq

    def generate_dataset(self, data, batch_size):
        dataset = TextDataset(data["x"], data["target"], self)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def generate_datasets(self, batch_size):
        datasets = {}
        for split in self.final_data:
            loader = self.generate_dataset(self.final_data[split], batch_size)
            datasets[split] = loader

        return datasets

class TextDataset(Dataset):
    def __init__(self, x, target, wrapper):
        self.x = x
        self.target = target
        self.start_token = wrapper.start_token
        self.vocab_size = wrapper.vocab_size

    def encode(self, tokens):
        mat = torch.nn.functional.one_hot(tokens, self.vocab_size)
        return mat

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx]).int()
        y_list = [self.start_token] + self.target[idx]
        comb_y = torch.tensor(y_list)
        y = comb_y[1:]
        prev_y = comb_y[:-1]
        return x, y, prev_y

class OpusDatasetWrapper(TextDatasetWrapper):
    name = "opus_books"
    version = "en-es"
    splits = ["train"]
    subset = "translation"

    def split_x_target(self, split):
        x = [s["es"] for s in split]
        target = [s["en"] for s in split]
        return x, target

    def trim_length(self, x, target):
        new_x = []
        new_target = []
        for xi, ti in zip(x, target):
            if len(xi) < self.x_length and len(ti) < self.target_length:
                new_x.append(xi)
                new_target.append(ti)
        return new_x, new_target

    def create_final_sets(self):
        x = self.encoded_data["train"]["x"]
        target = self.encoded_data["train"]["target"]

        valid_len = int(len(x) * 0.1)

        train_x = x[:-valid_len]
        valid_x = x[-valid_len:]
        train_target = target[:-valid_len]
        valid_target = target[-valid_len:]

        self.final_data = {"train": {"x": train_x, "target": train_target}, "validation": {"x": valid_x, "target": valid_target}}

class Opus100DatasetWrapper(OpusDatasetWrapper):
    name = "opus100"

class CNNDatasetWrapper(TextDatasetWrapper):
    name = "cnn_dailymail"
    version = "3.0.0"
    splits = ["train", "test", "validation"]
    subset = "highlights"
    x_length = 15
    target_length = 15

    def split_x_target(self, split):
        chunks = [s.split(" ") for s in split]
        chunks = [c for c in chunks if len(c) > self.x_length + self.target_length]
        x = [" ".join(c[:self.x_length]) for c in chunks]
        target = [" ".join(c[self.x_length:(self.x_length + self.target_length)]) for c in chunks]
        return x, target

    def trim_length(self, x, target):
        new_x = []
        new_target = []
        for xv, tr in zip(x, target):
            if len(xv) < self.x_length:
                continue

            xn = xv[:self.x_length]
            new_x.append(xn)
            xr = xv[self.x_length:]

            if len(xr) >= self.target_length:
                new_target.append(xr[:self.target_length])
            elif self.target_length > len(xr) > 0:
                new_target.append(xr + tr[:(self.target_length - len(xr))])
            else:
                new_target.append(tr[:self.target_length])
        return new_x, new_target