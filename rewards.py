import nltk
from nltk.translate.bleu_score import sentence_bleu
from easse.sari import sentence_sari


def bleu_reward(generated, gold, orig):
    generated = nltk.word_tokenize(generated)
    gold = nltk.word_tokenize(gold)
    bleu = sentence_bleu([gold], generated)
    reward = bleu + 0.001
    return reward


def sari_reward(generated, gold, orig):
    sari = sentence_sari(orig_sent=orig,
                         sys_sent=generated,
                         ref_sents=[gold])
    return sari
