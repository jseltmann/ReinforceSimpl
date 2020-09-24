import transformers as tr
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from random import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle

import load_data as ld
#from policy import Policy, GPT2Policy
#from rewards import bleu_reward, sari_reward, simple_english_reward

#import warnings
#warnings.simplefilter("ignore", message="Set")

def batchify(data, batch_size):
    """
    Split data into batches.

    Parameters
    ----------
    data : []
        List of data elements.
    batch_size : int
        Guess what.

    Return
    ------
    batched : [[]]
        List of batches.
    """
    batched = []
    curr_batch = []
    while data != []:
        curr_batch = data[:batch_size]
        batched.append(curr_batch)
        data = data[batch_size:]
    return batched

class LSTMModel(nn.Module):
    def __init__(self, tokenizer):
        super(LSTMModel, self).__init__()
        self.emb = nn.Embedding(tokenizer.vocab_size, 100)
        self.enc = nn.LSTM(100, 100)
        self.dec = nn.LSTM(100, 100)
        self.out = nn.Linear(100, tokenizer.vocab_size)

    def forward(self, input):
        device = input.device
        #input = input.transpose(0,1)
        hidden0 = torch.zeros(1, input.shape[1], 100).to(device)
        cell0 = torch.zeros(1, input.shape[1], 100).to(device)
        embedded = self.emb(input)
        _, (hidden, cell) = self.enc(embedded, (hidden0, cell0))
        dec_out, _ = self.dec(embedded, (hidden0, cell0))
        out = self.out(dec_out)
        #out = out.transpose(0,1)
        return out


class LSTMEncoder(nn.Module):
    def __init__(self, tokenizer, batch_size, hidden_size=100):
        super(LSTMEncoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(tokenizer.vocab_size, hidden_size)
        self.enc = nn.LSTM(hidden_size, hidden_size)
        #self.dec = nn.LSTM(100, 100)
        #self.out = nn.Linear(100, tokenizer.vocab_size)

    def forward(self, input, prev):
        #device = input.device
        #input = input.transpose(0,1)
        #hidden0 = torch.zeros(1, input.shape[1], 100).to(device)
        #cell0 = torch.zeros(1, input.shape[1], 100).to(device)
        embedded = self.emb(input)
        hidden, cell = prev
        out, (hidden, cell) = self.enc(embedded, (hidden, cell))
        #dec_out, _ = self.dec(embedded, (hidden0, cell0))
        #out = self.out(dec_out)
        #out = out.transpose(0,1)
        return out, (hidden, cell)

    def init_states(self, device, batch_size):
        hidden0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        cell0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

        return hidden0, cell0


class LSTMDecoder(nn.Module):
    def __init__(self, tokenizer, encoder, batch_size, hidden_size=100):
        super(LSTMDecoder, self).__init__()
        #self.emb = nn.Embedding(tokenizer.vocab_size, 100)
        self.enc = encoder
        #self.enc = nn.LSTM(100, 100)
        self.hidden_size = hidden_size
        self.dec = nn.LSTM(100, 100)
        self.out = nn.Linear(100, tokenizer.vocab_size)

    def forward(self, input, prev):
        #device = input.device
        #input = input.transpose(0,1)
        #hidden0 = torch.zeros(1, input.shape[1], 100).to(device)
        #cell0 = torch.zeros(1, input.shape[1], 100).to(device)
        embedded = self.enc.emb(input)
        hidden, cell = prev
        #_, (hidden, cell) = self.enc(embedded, (hidden0, cell0))
        dec_out, (hidden, cell) = self.dec(embedded, (hidden, cell))
        out = self.out(dec_out)
        #out = out.transpose(0,1)
        return out, (hidden, cell)

    def init_states(self, device, batch_size):
        hidden0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        cell0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

        return hidden0, cell0


#class LSTMEncoder(nn.Module):
#    def __init__(self, vocab, batch_size, hidden_size=100):
#        super(LSTMEncoder, self).__init__()
#        self.batch_size = batch_size
#        self.hidden_size = hidden_size
#        self.emb = nn.Embedding(len(vocab), hidden_size)
#        self.enc = nn.LSTM(hidden_size, hidden_size)
#        #self.dec = nn.LSTM(100, 100)
#        #self.out = nn.Linear(100, tokenizer.vocab_size)
#
#    def forward(self, input, prev):
#        #device = input.device
#        #input = input.transpose(0,1)
#        #hidden0 = torch.zeros(1, input.shape[1], 100).to(device)
#        #cell0 = torch.zeros(1, input.shape[1], 100).to(device)
#        embedded = self.emb(input)
#        hidden, cell = prev
#        out, (hidden, cell) = self.enc(embedded, (hidden, cell))
#        #dec_out, _ = self.dec(embedded, (hidden0, cell0))
#        #out = self.out(dec_out)
#        #out = out.transpose(0,1)
#        return out, (hidden, cell)
#
#    def init_states(self, device, batch_size):
#        hidden0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
#        cell0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
#
#        return hidden0, cell0
#
#
#class LSTMDecoder(nn.Module):
#    def __init__(self, vocab, encoder, batch_size, hidden_size=100):
#        super(LSTMDecoder, self).__init__()
#        #self.emb = nn.Embedding(tokenizer.vocab_size, 100)
#        self.enc = encoder
#        #self.enc = nn.LSTM(100, 100)
#        self.hidden_size = hidden_size
#        self.dec = nn.LSTM(100, 100)
#        self.out = nn.Linear(100, len(vocab))
#
#    def forward(self, input, prev):
#        #device = input.device
#        #input = input.transpose(0,1)
#        #hidden0 = torch.zeros(1, input.shape[1], 100).to(device)
#        #cell0 = torch.zeros(1, input.shape[1], 100).to(device)
#        embedded = self.enc.emb(input)
#        hidden, cell = prev
#        #_, (hidden, cell) = self.enc(embedded, (hidden0, cell0))
#        dec_out, (hidden, cell) = self.dec(embedded, (hidden, cell))
#        out = self.out(dec_out)
#        #out = out.transpose(0,1)
#        return out, (hidden, cell)
#
#    def init_states(self, device, batch_size):
#        hidden0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
#        cell0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
#
#        return hidden0, cell0



def train_supervised(data, vocab, model_save_path, device="cpu", epochs=1):
    """
    Train model using normal supervised training.

    Parameters
    ----------
    data : [(str,str)]
        Training pairs of normal and 
        simplified sentences or texts.
    model_save_path : str
        Path to save trained model to.
    device : str
        "cpu" or "cuda".
    """

    tokenizer = tr.BartTokenizer.from_pretrained("facebook/bart-base")
    
    writer = SummaryWriter("/data/tensorboard/supervised/lstm")

    batch_size = 8
    batches = batchify(data, batch_size)

    max_len = 70
    data_ids = []
    for batch in tqdm(batches):
        normal_batch = [pair[0] for pair in batch]
        simple_batch = [pair[1] for pair in batch]
        #normal_words = [s.split() for s in normal_batch]
        #simple_words = [s.split() for s in simple_batch]
        #max_len = max([len(sent) for sent in normal_words] + [len(sent) for sent in simple_words])
        #normal_inds = []
        #for word_seq in normal_words:
        #    pad_num = max_len - len(word_seq)
        #    word_seq = ["[BOS]"] + word_seq + ["[EOS]"] + pad_num * ["[PAD]"]
        #    ind_seq = []
        #    for word in word_seq:
        #        if word.lower() in vocab:
        #            ind_seq.append(vocab[word])
        #        else:
        #            ind_seq.append(vocab["[UNK]"])
        #    normal_inds.append(ind_seq)
        #normal_ids = torch.Tensor(normal_inds)
        #
        #simple_inds = []
        #for word_seq in simple_words:
        #    pad_num = max_len - len(word_seq)
        #    word_seq = ["[BOS]"] + word_seq + ["[EOS]"] + pad_num * ["[PAD]"]
        #    ind_seq = []
        #    for word in word_seq:
        #        if word.lower() in vocab:
        #            ind_seq.append(vocab[word])
        #        else:
        #            ind_seq.append(vocab["[UNK]"])
        #    simple_inds.append(ind_seq)
        #simple_ids = torch.Tensor(simple_inds)
        normal_ids = tokenizer(normal_batch, padding='max_length', return_tensors='pt', max_length=max_len, truncation=True)
        simple_ids = tokenizer(simple_batch, padding='max_length', return_tensors='pt', max_length=max_len, truncation=True)
        data_ids.append((normal_ids, simple_ids))

    #model = LSTMModel(tokenizer)
    encoder = LSTMEncoder(tokenizer, batch_size)
    encoder.to(device)
    decoder = LSTMDecoder(tokenizer, encoder, batch_size)
    decoder.to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean')
    to_optimize = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(to_optimize, lr=1e-4)

    i = 0
    #for epoch in range(epochs):
    for epoch in tqdm(range(epochs)):
        #print(epoch)
        #for batch in tqdm(data_ids):
        for batch in data_ids:
            normal_batch = batch[0]["input_ids"].to(device)
            simple_batch = batch[1]["input_ids"].to(device)
            #print(normal_batch[0].shape)
            #print(normal_batch[0])
            #8 / 0

            #generated = model(normal_batch["input_ids"])
            #generated = generated.transpose(1,2)

            batch_size = normal_batch.shape[0]

            normal_batch = normal_batch.transpose(0,1)
            hidden, cell = encoder.init_states(device, batch_size)
            for tok_batch in normal_batch:
                tok_batch = tok_batch.unsqueeze(0)
                _, (hidden, cell) = encoder(tok_batch, (hidden, cell))

            _, cell = decoder.init_states(device, batch_size)
            simple_batch = simple_batch.transpose(0,1)
            outs = []
            for tok_batch in simple_batch[:-1]:
                # use elements from simple_batch since we're doing Teacher Forcing
                tok_batch = tok_batch.unsqueeze(0)
                out, (hidden, cell) = decoder(tok_batch, (hidden, cell))
                outs.append(out)
            generated = torch.stack(outs)
            generated = generated.squeeze(1).transpose(0,1).transpose(1,2)

            simple_batch = simple_batch[1:]
            simple_batch = simple_batch.transpose(0,1)
            loss = loss_fn(generated, simple_batch)
            writer.add_scalar("Loss", loss.item(), i)
            i += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #torch.save(encoder.state_dict(), model_save_path+".enc"+str(epoch))
        #torch.save(decoder.state_dict(), model_save_path+".dec"+str(epoch))

    #torch.save(model.state_dict(), model_save_path)
    torch.save(encoder.state_dict(), model_save_path+".enc")
    torch.save(decoder.state_dict(), model_save_path+".dec")


if __name__ == "__main__":
    #data = ld.load_wiki_sents("/data/data/wiki/sent_aligned_split/train")
    data = ld.load_newsela_sents("/data/data/newsela_tok/V0V2/train")[:1]
    vpath = "/data/data/newsela_tok/vocab.pickle"
    with open(vpath, "rb") as vfile:
        vocab = pickle.load(vfile)
    #train_supervised(data, "/data/tuned_models/supervised/lstm_newsela_teacherf.pt", device="cuda", epochs=2000)
    train_supervised(data, vocab, "/data/tuned_models/supervised/lstm_newsela_teacherf.pt", device="cuda", epochs=2000)
