import nltk
from nltk.translate.bleu_score import sentence_bleu


def bleu_reward(generated, gold):
    generated = nltk.word_tokenize(generated)
    gold = nltk.word_tokenize(gold)
    bleu = sentence_bleu([gold], generated)
    reward = bleu + 0.001
    return bleu
