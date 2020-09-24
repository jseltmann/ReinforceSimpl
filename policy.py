import torch
import torch.nn as nn
import transformers as tr
import nltk


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.model = tr.BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = tr.BartTokenizer.from_pretrained("facebook/bart-base")

    def forward(self, x, device="cpu", max_len=70):
        x = x.to(device)
        generated, probs = self.model.generate(x, num_beams=1, return_probs=True, max_length=max_len)
        return generated, probs

    def tokenize(self, batch):
        toks = self.tokenizer(batch, padding=True, return_tensors='pt')["input_ids"]
        return toks

    def decode(self, sampled):
        sent = self.tokenizer.batch_decode(sampled, skip_special_tokens=True)
        return sent


class GPT2Policy(nn.Module):
    def __init__(self):
        super(GPT2Policy, self).__init__()
        self.model = tr.GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = tr.GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, x, device="cpu", max_len=70):
        x = x.to(device)
        generated, probs = self.model.generate(x, num_beams=1, return_probs=True, max_length=max_len)
        return generated, probs

    def tokenize(self, batch, max_len=60):
        toks = self.tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt')["input_ids"]
        return toks

    def decode(self, sampled):
        sent = self.tokenizer.batch_decode(sampled, skip_special_tokens=True)
        return sent



class LSTMPolicy(nn.Module):
    def __init__(self, hidden_size=100):
        super(LSTMPolicy, self).__init__()
        #self.model = tr.BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = tr.BartTokenizer.from_pretrained("facebook/bart-base")
        self.embeddings = nn.Embedding(tokenizer.vocab_size(), hidden_size)
        self.enc = nn.LSTM(hidden_size, hidden_size)
        self.dec = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, tokenizer.vocab_size)
        self.max_len = 70

    def forward(self, input, prev):
        embedded = self.emb(input)
        hidden, cell = prev
        out, (hidden, cell) = self.enc(embedded, (hidden, cell))
        #x = x.to(device)
        #generated, probs = self.model.generate(x, num_beams=1, return_probs=True, max_length=max_len)
        #return generated, probs

    def tokenize(self, batch):
        toks = self.tokenizer(batch, padding=True, return_tensors='pt')["input_ids"]
        return toks

    def decode(self, sampled):
        sent = self.tokenizer.batch_decode(sampled, skip_special_tokens=True)
        return sent

    def encode_seq(self, input):
        batch_size = input.shape[0]
        input = input.transpose(0,1)
        emb = self.embeddings(input)
        hidden0 = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        cell0 = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        _, (hidden, cell) = self.enc(emb, (self.hidden0, self.cell0))
        return hidden

    def decode_seq(self, enc_hidden, input=None):
        device = enc_hidden.device
        batch_size = enc_hidden.shape[1]
        hidden = enc_hidden
        cell = torch.zeros(1, batch_size, self.hidden_size)
        if input is None:
            # don't do teacher forcing
            curr_tok_id = torch.Tensor([self.tokenizer.bos_token_id]).to(device)
            out_probs = []
            out_ids = []
            for i in range(self.max_len):
                out, (hidden, cell) = self.decoder(curr_tok_id, (hidden, cell))
                out_probs.append(out)
                curr_tok_id = torch.argmax(out, dim=2)
                out_ids.append(curr_tok_id)
        else:
            # do teacher forcing
            out_probs = []
            out_ids = []
            input = input.transpose(0,1)
            for tok_batch in input[:-1]:
                out, (hidden, cell) = self.decoder(tok_batch, (hidden, cell))


        #batch_size = input.shape[0]
        #input = input.transpose(0,1)
        #emb = self.embeddings(input)
        #hidden0 = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        #cell0 = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        #_, (hidden, cell) = self.enc(emb, (self.hidden0, self.cell0))
        #return hidden
