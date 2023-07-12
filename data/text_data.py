from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast
import datasets
import os
import re


def chunk_tokens_and_ids(examples, chunk_size, ids_key, tokens_key):
    chunked_ids = []
    chunked_tokens = []
    for id, token in zip(examples[ids_key], examples[tokens_key]):
        chunked_ids += [id[i:i + chunk_size] for i in range(0, len(id), chunk_size)]
        chunked_tokens += [token[i:i + chunk_size] for i in range(0, len(token), chunk_size)]
    return {tokens_key: chunked_tokens, ids_key: chunked_ids}


def get_tokens_and_ids(examples, tokenizer, text_key, ids_key, tokens_key, padding=False):
    tokens = []
    ids = []
    for example in examples[text_key]:
        token = tokenizer.tokenize(example, padding=padding)
        id = tokenizer.convert_tokens_to_ids(token)
        tokens.append(token)
        ids.append(id)
    return {tokens_key: tokens, ids_key: ids}


def calc_token_ratio(examples, ids_key, data_key):
    ratios = []
    for id, data in zip(examples[ids_key], examples[data_key]):
        if len(data) == 0:
            ratios.append(0)
        else:
            ratios.append(len(id) / len(data))
    return {"token_ratio": ratios}


class DatasetWrapper:
    dataset_name = None
    data_config = None
    data_key = "text"
    ids_key = "input_ids"
    tokens_key = "tokens"
    max_token_ratio = .33
    run_combine_func = False
    run_split_func = False
    run_chunking = True

    def __init__(self, download_split="train", model_max_length=512, processes=None, download_split_pct=None, tokenizer_vocab=5000, min_token_freq=2):
        self.download_split_pct = download_split_pct
        self.tokenizer_filename = f"{self.dataset_name}_{download_split_pct}_tokenizer"
        self.model_max_length = model_max_length
        self.processes = processes
        self.download_split = download_split
        self.tokenizer_vocab = tokenizer_vocab
        self.min_token_freq = min_token_freq
        self.tokenizer = None

    def dataset_info(self):
        data = datasets.load_dataset_builder(self.dataset_name, self.data_config)
        print(data.info.description)
        print(data.info.splits)
        print(data.info.features)

    def process_dataset(self):
        data = self.load_dataset()
        if self.run_combine_func:
            data = data.map(lambda x: self.combine_func(x), batched=True, remove_columns=data.column_names, num_proc=self.processes, desc="Combining")

        self.tokenizer = self.get_tokenizer(data[self.data_key])
        tokenized = self.tokenize_dataset(data, self.tokenizer)
        tokenized = self.filter_tokenized_text(tokenized)
        if self.run_chunking:
            tokenized = self.chunk_tokens(tokenized)
        if self.run_split_func:
            tokenized = tokenized.map(lambda x: self.split_func(x), batched=True, remove_columns=tokenized.column_names, num_proc=self.processes, desc="Splitting")
        tokenized = tokenized.with_format("torch")
        return tokenized

    def combine_func(self, examples):
        """A function that combines examples in the dataset."""
        raise NotImplementedError

    def split_func(self, examples):
        """A function that splits examples in the dataset."""
        raise NotImplementedError

    def load_dataset(self):
        split_str = self.download_split
        if self.download_split_pct:
            split_str = f"{self.download_split}[:{self.download_split_pct}]"
        if not self.data_config:
            data = datasets.load_dataset(self.dataset_name, split=split_str, num_proc=self.processes)
        else:
            data = datasets.load_dataset(self.dataset_name, self.data_config, split=split_str, num_proc=self.processes)
        return data

    def tokenize_dataset(self, data, tokenizer):
        data = data.map(lambda examples: get_tokens_and_ids(examples, tokenizer, self.data_key, self.ids_key, self.tokens_key), batched=True, num_proc=self.processes, desc="Tokenizing")
        return data

    def filter_tokenized_text(self, data):
        data = data.map(lambda x: calc_token_ratio(x, self.ids_key, self.data_key), batched=True, num_proc=self.processes, desc="Filtering")
        data = data.filter(lambda x: 0 < x["token_ratio"] < self.max_token_ratio, num_proc=self.processes, desc="Filtering")
        return data

    def chunk_tokens(self, data):
        data_chunks = data.map(lambda examples: chunk_tokens_and_ids(examples, self.model_max_length, self.ids_key, self.tokens_key), batched=True, remove_columns=data.column_names, num_proc=self.processes, desc="Chunking")
        return data_chunks

    def get_tokenizer(self, data=None):
        if os.path.exists(self.tokenizer_filename):
            tokenizer = self.load_tokenizer(self.tokenizer_filename)
        else:
            tokenizer = self.train_tokenizer(data, self.tokenizer_filename)
        return tokenizer

    def train_tokenizer(self, text, save_file):
        special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
        sptokenizer = SentencePieceBPETokenizer()
        sptokenizer.train_from_iterator(
            text,
            vocab_size=self.tokenizer_vocab,
            min_frequency=self.min_token_freq,
            show_progress=True,
            special_tokens=special_tokens
        )

        tokenizer = PreTrainedTokenizerFast(tokenizer_object=sptokenizer, special_tokens=special_tokens)
        for attr, token in zip(["pad_token", "bos_token", "eos_token", "unk_token"], special_tokens):
            setattr(tokenizer, attr, token)
            setattr(tokenizer, f"{attr}_id", sptokenizer.token_to_id(token))
        tokenizer.save_pretrained(save_file)
        return tokenizer

    def load_tokenizer(self, tokenizer_file):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
        return tokenizer

    def decode_ids(self, ids):
        # Add a batch dimension if needed
        if len(ids.shape) == 1:
            ids = ids.unsqueeze(0)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)


class WikiTextDataset(DatasetWrapper):
    dataset_name = "wikitext"
    data_config = "wikitext-103-v1"
    data_key = "text"
    max_token_ratio = .33
    run_combine_func = True
    run_split_func = True

    def combine_func(self, examples):
        entries = []
        entry = ""
        for sentence in examples[self.data_key]:
            if re.match("^ \= \w", sentence):
                if entry:
                    entries.append(entry)
                entry = ""
            else:
                entry += sentence
        entries.append(entry)
        return {self.data_key: entries}

    def split_func(self, examples):
        all_ids = []
        all_tokens = []
        for ids, tokens in zip(examples[self.ids_key], examples[self.tokens_key]):
            # Filter out sequences that are too long or too short
            if len(ids) == self.model_max_length:
                all_ids.append(ids)
                all_tokens.append(tokens)
        return {self.ids_key: all_ids, self.tokens_key: all_tokens}


class OpusBooksDataset(DatasetWrapper):
    dataset_name = "opus_books"
    data_config = "en-es"
    en_key = "en"
    es_key = "es"
    data_key = "translation"
    max_token_ratio = .33
    run_combine_func = True
    run_split_func = True
    run_chunking = False
    max_length = 50
    min_length = 10

    def combine_func(self, examples):
        entries = []
        for sentence in examples[self.data_key]:
            en = sentence[self.en_key]
            es = sentence[self.es_key]
            comb = f"{en}<s>{es}"
            entries.append(comb)
        return {self.data_key: entries}

    def split_func(self, examples):
        en_ids = []
        es_ids = []
        en_lens = []
        for ids, tokens in zip(examples[self.ids_key], examples[self.tokens_key]):
            split_ind = tokens.index("<s>")
            en_id = ids[:split_ind]
            es_id = ids[split_ind+1:]
            # Filter out sequences that are too long or too short
            if self.min_length <= len(en_id) <= self.max_length and self.min_length <= len(es_id) <= self.max_length:
                en_lens.append(len(en_id))
                en_ids.append(en_id)
                es_ids.append(es_id)
        return {"en_ids": en_ids, "es_ids": es_ids, "en_lens": en_lens}