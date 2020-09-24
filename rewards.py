import nltk
from nltk.translate.bleu_score import sentence_bleu
from easse.sari import sentence_sari
import string
import stanza
from stanza.server import CoreNLPClient
from sentence_transformers import SentenceTransformer, util

from word_list import be_combined

nlp = stanza.Pipeline('en')
client = CoreNLPClient(annotators=['parse'], timeout=30000, memory='16G')
sts_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


# tregexes taken from the L2 Syntactic Complexity Analyzer
# http://www.personal.psu.edu/xxl13/downloads/l2sca.html
clause_tregex = 'S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]'
sent_tregex='ROOT'



#def close_client():
#    client.close()

def _length_penalty(generated, gold):
    frac = len(generated) / len(gold)
    len_penalty = 1 - frac
    if len_penalty < 0.25:
        len_penalty = 0.25
    return len_penalty


def _length_penalty_orig(generated, orig):
    frac = len(generated) / len(gold)
    len_penalty = 1 - frac
    return len_penalty


def bleu_reward(generated, gold, orig):
    generated = nltk.word_tokenize(generated)
    gold = nltk.word_tokenize(gold)
    bleu = sentence_bleu([gold], generated)
    len_penalty = _length_penalty(generated, gold)
    reward = len_penalty * (bleu + 0.001)
    return reward


def sari_reward(generated, gold, orig):
    sari = sentence_sari(orig_sent=orig,
                         sys_sent=generated,
                         ref_sents=[gold])
    return sari


def simple_english_reward(generated, gold, orig):
    # reward based on simple wikipedia instructions
    # https://simple.wikipedia.org/wiki/Wikipedia:How_to_write_Simple_English_pages

    try:
        doc = nlp(generated)
        
        # find fraction of words contained in BE 1500 list
        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        lemmas = [l for l in lemmas if l not in string.punctuation]
        lemmas_in_vocab = [l for l in lemmas if l in be_combined]
        if len(lemmas) == 0:
            frac_in_vocab = 0
        else:
            frac_in_vocab = len(lemmas_in_vocab) / len(lemmas)

        # find syntactic complexity
        clause_matches = client.tregex(generated, clause_tregex)
        num_clauses = 0
        num_sents = 0
        for sent_matches in clause_matches["sentences"]:
            num_clauses += len(sent_matches)
            num_sents += 1
        if num_clauses == 0 or len(lemmas) == 0:
            mlc = 1000000000000000000 # just some very high number
        else:
            mlc = len(lemmas) / num_clauses # simple= 16.6, 9.9; normal= 19.1, 11.5
        if num_sents == 0 or num_clauses == 0:
            c_s = 1000000000000000000
        else:
            c_s = num_clauses / num_sents # reward: 1/c_s
        syntax_reward = 0.5 * 1 / mlc + 0.5 * 1 / c_s

        # find sematic similarity
        emb1 = sts_model.encode([orig])
        emb2 = sts_model.encode([generated])
        sim = util.pytorch_cos_sim(emb1, emb2).item()

        comb_reward = frac_in_vocab + syntax_reward + sim

        return comb_reward
    except Exception as e:
        return 0
