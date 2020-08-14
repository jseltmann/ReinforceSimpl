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

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, device="cpu"):
        #x2 = self.model.generate(x, max_length=30, return_probs=True)
        input_ids = x["input_ids"].to(device)
        attention_mask = x["attention_mask"].to(device)
        logits, _ = self.model(input_ids, attention_mask=attention_mask)
        probs = self.softmax(logits)
        return probs

    def sample_greedy(self, probs):
        max_probs_batch, max_inds_batch = torch.max(probs, dim=2)
        sents = []
        for max_inds in max_inds_batch:
            sent = self.tokenizer.decode(max_inds, skip_special_tokens=True)
            sents.append(sent)

        sent_probs = torch.prod(max_probs_batch, dim=1)

        return sents, sent_probs

    def tokenize(self, batch):
        toks = self.tokenizer(batch, padding=True, return_tensors='pt')#['input_ids']
        return toks

    def decode(self, sampled):
        sent = self.tokenizer.decode(sampled, skip_special_tokens=True)
        return sent

    def find_normal_word_inds(self, generated_ids):
        """
        Find which word indices in a generated sequence 
        are words instead of special tokens.
        """
        special_ids = set()
        special_ids.add(self.tokenizer.bos_token_id)
        special_ids.add(self.tokenizer.eos_token_id)
        special_ids.add(self.tokenizer.unk_token_id)
        special_ids.add(self.tokenizer.sep_token_id)
        special_ids.add(self.tokenizer.pad_token_id)
        special_ids.add(self.tokenizer.cls_token_id)
        special_ids.add(self.tokenizer.mask_token_id)
        special_ids.update(self.tokenizer.additional_special_tokens)

        normal_inds = [i for i, gid in enumerate(generated_ids[0]) \
                if not int(gid) in special_ids]

        return normal_inds
        
