import torch
import torch.nn as nn
import transformers as tr
import nltk


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.model = tr.BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.softmax = nn.Softmax(dim=2)
        self.tokenizer = tr.BartTokenizer.from_pretrained("facebook/bart-base")

    def forward(self, x, device="cpu"):
        x = x.to(device)
        generated, probs = self.model.generate(x, num_beams=1, return_probs=True)
        return generated, probs

    def sample_greedy(self, probs):
        max_probs_batch, max_inds_batch = torch.max(probs, dim=2)
        sents = []
        for max_inds in max_inds_batch:
            sent = self.tokenizer.decode(max_inds, skip_special_tokens=True)
            sents.append(sent)

        sent_probs = torch.prod(max_probs_batch, dim=1)

        return sents, sent_probs

    def tokenize(self, batch):
        toks = self.tokenizer(batch, padding=True, return_tensors='pt')["input_ids"]
        return toks

    def decode(self, sampled):
        sent = self.tokenizer.batch_decode(sampled, skip_special_tokens=True)
        return sent
