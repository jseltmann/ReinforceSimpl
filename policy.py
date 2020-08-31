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
