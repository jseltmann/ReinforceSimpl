import transformers as tr
import torch
from tqdm import tqdm
import math

import load_data as ld
from policy import Policy
from rewards import bleu_reward


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


def train_base(data, reward_fn, model_save_path, device="cpu"):
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
    device : str
        "cpu" or "cuda".
    """

    policy = Policy()
    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    batches = batchify(data, 8)

    loss0 = torch.tensor(0.).to(device)

    for batch in tqdm(batches):
        normal_batch = [pair[0] for pair in batch]
        simple_batch = [pair[1] for pair in batch]

        inputs_prepared = policy.tokenize(normal_batch)
        sampled_sents, sent_probs = policy(inputs_prepared, device=device)
        sampled_sents = policy.decode(sampled_sents)

        rewards = []
        for sampled, simple in zip(sampled_sents, simple_batch):
            reward = reward_fn(sampled, simple)
            rewards.append(reward)

        loss = loss0.detach().clone()
        for reward, sent_prob in zip(rewards, sent_probs):
            loss -= reward * torch.log(sent_prob)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(policy.state_dict(), model_save_path)


data = ld.load_wiki_sents("/data/wiki/sent_aligned_split/train")
train_base(data, bleu_reward, "/data/tuned_models/bleu/test/test_bleu.pt", device="cuda")
