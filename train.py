import transformers as tr
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from random import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import load_data as ld
from policy import Policy, GPT2Policy
from rewards import bleu_reward, sari_reward, simple_english_reward

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


def train_base(data, reward_fn, model_save_path, logdir, device="cpu", epochs=1):
    """
    Train model.

    Parameters
    ----------
    data : [(str,str)]
        Training pairs of normal and 
        simplified sentences or texts.
    reward_fn : function
        Function that takes the produced sentence and
        the gold sentence and produces an reward score.
    model_save_path : str
        Path to save trained model to.
    logdir : str
        Path to save tensorboard logs to.
    device : str
        "cpu" or "cuda".
    """

    policy = Policy()
    policy.train()
    policy.to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5)
    writer = SummaryWriter(logdir)

    batches = batchify(data, 8)

    loss0 = torch.tensor(0.).to(device)

    i = 0
    for epoch in range(epochs):
        for batch in tqdm(batches):
            normal_batch = [pair[0] for pair in batch]
            simple_batch = [pair[1] for pair in batch]

            inputs_prepared = policy.tokenize(normal_batch)
            sampled_sents, sent_probs = policy(inputs_prepared, device=device)
            sampled_sents = policy.decode(sampled_sents)

            rewards = []
            for sampled, simple, orig in zip(sampled_sents, simple_batch, normal_batch):
                reward = reward_fn(sampled, simple, orig)
                rewards.append(reward)
            writer.add_scalar("Reward mean", np.mean(rewards), i)
            writer.add_scalar("Reward std dev", np.std(rewards), i)

            loss = loss0.detach().clone()
            for reward, sent_prob in zip(rewards, sent_probs):
                loss -= reward * torch.log(sent_prob)
            writer.add_scalar("Loss", loss.item(), i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
        torch.save(policy.state_dict(), model_save_path+str(epoch))

    torch.save(policy.state_dict(), model_save_path)
    #close_client()


def train_supervised(data, model_save_path, device="cpu", epochs=1):
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
    #tokenizer = tr.BertTokenizer.from_pretrained("bert-base-uncased")
    #tokenizer = tr.GPT2Tokenizer.from_pretrained("gpt2")
    #tokenizer.pad_token = tokenizer.eos_token
    writer = SummaryWriter("/data/tensorboard/supervised/30ep")

    batches = batchify(data, 8)

    data_ids = []
    for batch in tqdm(batches):
        normal_batch = [pair[0] for pair in batch]
        simple_batch = [pair[1] for pair in batch]
        normal_ids = tokenizer(normal_batch, padding='max_length', return_tensors='pt', max_length=70, truncation=True)
        simple_ids = tokenizer(simple_batch, padding='max_length', return_tensors='pt', max_length=70, truncation=True)
        data_ids.append((normal_ids, simple_ids))

    model = tr.BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    #model = tr.GPT2LMHeadModel.from_pretrained("gpt2")
    #config = tr.GPT2Config.from_pretrained("gpt2")
    #model = tr.GPT2LMHeadModel(config)
    #config = tr.BartConfig.from_pretrained("facebook/bart-base")
    #model = tr.BartForConditionalGeneration(config)
    model.to(device)
    model.train()
    #optimizer = tr.AdamW(model.parameters(), lr=5e-5)
    optimizer = tr.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean')

    i = 0
    for epoch in tqdm(range(epochs)):
        #print(epoch)
        #for batch in tqdm(data_ids):
        for batch in data_ids:
            normal_batch = batch[0].to(device)
            simple_batch = batch[1].to(device)

            outputs = model(**normal_batch)
            generated = outputs[0]
            sums = generated.sum(2)
            generated = generated.permute(0,2,1)
            loss = loss_fn(generated, simple_batch["input_ids"])
            writer.add_scalar("Loss", loss.item(), i)
            i += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), model_save_path+str(epoch))

    torch.save(model.state_dict(), model_save_path)


#data = ld.load_wiki_sents("/data/data/wiki/sent_aligned_split/train")
data = ld.load_newsela_sents("/data/data/newsela/V0V1/train")
#train_base(data, simple_english_reward, "/data/tuned_models/bleu/bleu_newsela.pt", "/data/tensorboard/bleu/bleu_newsela", device="cuda", epochs=10)
#train_base(data, simple_english_reward, "/data/tuned_models/ser/bart_ser.pt", "/data/tensorboard/ser/ser_test", device="cuda", epochs=1)
train_supervised(data, "/data/tuned_models/supervised/bart_newsela.pt", device="cuda", epochs=20)
