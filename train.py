import transformers as tr
import torch
from tqdm import tqdm

import load_data as ld
from policy import Policy
from rewards import bleu_reward


def train_base(data, reward_fn, model_save_path):
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
    """

    policy = Policy()
    #policy.to("cuda")

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    
    for normal, simple in tqdm(data[:5000]):
        input_ids = policy.tokenize(normal)
        #logits, sampled = policy(toks)
        #words = policy.decode(sampled)
        #(logits, _), generated = policy(input_ids)
        #generated_ids, probs = policy(input_ids)
        probs = policy(input_ids)
        sampled_sent, sent_prob = policy.sample_greedy(probs)
        #generated_words = policy.decode(generated_ids[0])
        reward = reward_fn(sampled_sent, simple)


        loss = -torch.log(reward * sent_prob)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(policy.state_dict(), model_save_path)



data = ld.load_wiki_sents("/data/wiki/sent_aligned_split/train")
train_base(data, bleu_reward, "/data/tuned_models/bleu/test/test_bleu_5000.pt")
