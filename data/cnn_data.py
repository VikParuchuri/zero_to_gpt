from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import math
from torchtext.data.functional import generate_sp_model
from torchtext.data.functional import load_sp_model, sentencepiece_numericalizer
import numpy as np
import sentencepiece as spm

MODEL_PREFIX = "cnn"
MODEL_FILE = f"{MODEL_PREFIX}.model"
TOKENS_FILE = "tokens.txt"

def chunk_text(texts, chunk_length):
    return [(t[:chunk_length], t[chunk_length:(2 * chunk_length)]) for t in texts if (chunk_length * 2) <= len(t)]

def decode_ids(ids, sp_base, vocab_size):
    if isinstance(ids, torch.Tensor):
        ids = [int(i) for i in list(ids.numpy()) if int(i) < vocab_size]
    return sp_base.decode(ids)

def decode_batch(id_tensor, vocab_size):
    sp_base = spm.SentencePieceProcessor(model_file=MODEL_FILE)
    decoded = []
    for i in range(id_tensor.shape[0]):
        decoded.append(decode_ids(id_tensor[i, :], sp_base, vocab_size))
    return decoded

def encode(tokens, vocab_size):
    mat = np.zeros((len(tokens), vocab_size))
    for i in range(len(tokens)):
        mat[i, tokens[i]] = 1
    return mat

class CNNDataset(Dataset):
    def __init__(self, data, start_token, stop_token, device, vocab_size):
        self.dataset = data
        self.device = device
        self.start_token = start_token
        self.stop_token = stop_token
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = torch.tensor(self.dataset[idx][0]).int()
        y_list = self.dataset[idx][1]
        y = torch.tensor(encode(y_list, self.vocab_size)).float()
        return x.to(self.device), y.to(self.device)

def load_data_list(data, key, text_length, chunk_length):
    selection = data[key]["highlights"][:text_length]
    return selection

def generate_data(train_length=1000, valid_length=250, test_length=250, vocab_size=1000, chunk_length=12, batch_size=8, device="cpu", retrain_tokenizer=True):
    stop_token = vocab_size
    start_token = vocab_size + 1
    chunk_length = 12

    # Load from Huggingface datasets module
    data = load_dataset("cnn_dailymail", "3.0.0")
    train = load_data_list(data, "train", train_length, chunk_length)
    valid = load_data_list(data, "validation", valid_length, chunk_length)
    test = load_data_list(data, "test", test_length, chunk_length)

    if retrain_tokenizer:
        with open(TOKENS_FILE, "w+") as f:
            f.write("\n".join(train) + "\n".join(valid) + "\n".join(test))

        generate_sp_model(TOKENS_FILE, vocab_size=vocab_size, model_prefix=MODEL_PREFIX)
    sp_model = load_sp_model(MODEL_FILE)
    encoding_generator = sentencepiece_numericalizer(sp_model)

    def generate_dataset(data):
        ids = list(encoding_generator(data))
        ids = chunk_text(ids, chunk_length)
        dataset = CNNDataset(ids, start_token, stop_token, device, vocab_size+2)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

    train_loader = generate_dataset(train)
    valid_loader = generate_dataset(valid)
    test_loader = generate_dataset(test)

    return train_loader, valid_loader, test_loader