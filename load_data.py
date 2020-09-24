import os
import random
import nltk
import collections as col
import pickle

random.seed(42)

def split_wiki_sent(data_dir, split_dir):
    """
    Split the sentence-aligned wikipedia set into train and test data.

    Parameters
    ----------
    data_dir : str
        Directory containing original files.
    split_dir : str
        Directory to save split corpus to.
    """

    normal_path = os.path.join(data_dir, "normal.aligned")
    simple_path = os.path.join(data_dir, "simple.aligned")

    normal_lines = []
    simple_lines = []
    skip_inds = set()
    with open(normal_path) as nfile:
        #normal_lines = nfile.readlines()
        for i, line in enumerate(nfile):
            _, _, sent = line.split("\t")
            sent = sent.split()
            if len(sent) > 80:
                skip_inds.add(i)
            normal_lines.append(line)
    with open(simple_path) as sfile:
        #simple_lines = sfile.readlines()
        for i, line in enumerate(sfile):
            _, _, sent = line.split("\t")
            sent = sent.split()
            if len(sent) > 80:
                skip_inds.add(i)
            simple_lines.append(line)

    zipped = list(zip(normal_lines, simple_lines))
    zipped = [p for i, p in enumerate(zipped) if i not in skip_inds]

    random.shuffle(zipped)

    cutoff = int(0.9 * len(zipped))
    train = zipped[:cutoff]
    test = zipped[cutoff:]

    normal_path = os.path.join(split_dir, "train/normal.aligned")
    simple_path = os.path.join(split_dir, "train/simple.aligned")
    with open(normal_path, "w") as nfile, open(simple_path, "w") as sfile:
        for nline, sline in train:
            nfile.write(nline)
            sfile.write(sline)

    normal_path = os.path.join(split_dir, "test/normal.aligned")
    simple_path = os.path.join(split_dir, "test/simple.aligned")
    with open(normal_path, "w") as nfile, open(simple_path, "w") as sfile:
        for nline, sline in test:
            nfile.write(nline)
            sfile.write(sline)

#split_wiki_sent("/data/sentence-aligned.v2",
#                "/data/wiki/sent_aligned_split")


def split_newsela_sents(newsela_path, split_dir):
    """
    Split the sentence-aligned newsela set into train and test data.
    Append difficulty information about original
    and target sentence to original sentence.

    Parameters
    ----------
    newsela_path : str
        Path to txt file containing aligned newsela sents.
    split_dir : str
        Directory to save split corpus to.
    """

    pairs = []
    with open(newsela_path) as newsela_file:
        for line in newsela_file:
            entries = line.split("\t")
            v_higher = entries[1]
            v_lower = entries[2]
            higher = v_higher + " " + entries[3].strip() + " " + v_lower
            lower = entries[4].strip()
            pairs.append((higher, lower))

    random.shuffle(pairs)

    cutoff = int(0.9 * len(pairs))
    train = pairs[:cutoff]
    test = pairs[cutoff:]

    normal_path = os.path.join(split_dir, "train/normal.aligned")
    simple_path = os.path.join(split_dir, "train/simple.aligned")
    with open(normal_path, "w") as nfile, open(simple_path, "w") as sfile:
        for nsent, ssent in test:
            nfile.write(nsent + "\n")
            sfile.write(ssent + "\n")

    normal_path = os.path.join(split_dir, "test/normal.aligned")
    simple_path = os.path.join(split_dir, "test/simple.aligned")
    with open(normal_path, "w") as nfile, open(simple_path, "w") as sfile:
        for nsent, ssent in test:
            nfile.write(nsent + "\n")
            sfile.write(ssent + "\n")

split_newsela_sents("/data/data/newsela_article_corpus_2016-01-29/newsela_data_share-20150302/newsela_articles_20150302.aligned.sents.txt",
                             "/data/data/newsela/all_versions")


def split_newsela_sents_versions(newsela_path, split_dir, do_tok=False, voc_size=50000):
    """
    Split the sentence-aligned newsela set into train and test data.
    Create one file for each combination of language levels.

    Parameters
    ----------
    newsela_path : str
        Path to txt file containing aligned newsela sents.
    split_dir : str
        Directory to save split corpus to.
    do_tok : Bool
        Whether or not to pre-tokenized and lower-case the sentences.
    voc_size : int
        Size of vocabulary to use.
    """

    version_pairs = col.defaultdict(list)
    words = col.defaultdict(int)

    with open(newsela_path) as newsela_file:
        for line in newsela_file:
            entries = line.split("\t")
            vs = entries[1] + entries[2]
            higher = entries[3].strip()
            lower = entries[4].strip()
            for w in nltk.word_tokenize(higher):
                words[w.lower()] += 1
            for w in nltk.word_tokenize(lower):
                words[w.lower()] += 1
            version_pairs[vs].append((higher, lower))

    words = list(words.items())
    words = sorted(words, key=lambda x: x[1])
    words = words[:voc_size-1]
    words.append("[BOS]")
    words.append("[EOS]")
    words.append("[PAD]")
    words.append("[UNK]")
    vocab = dict()
    for i, w in enumerate(words):
        vocab[w] = i

    for vs in version_pairs:
        comb_dir = os.path.join(split_dir, vs)
        if not os.path.exists(comb_dir):
            os.makedirs(comb_dir + "/train")
            os.makedirs(comb_dir + "/test")
        pairs = version_pairs[vs]
        random.shuffle(pairs)

        cutoff = int(0.9 * len(pairs))
        train = pairs[:cutoff]
        test = pairs[cutoff:]

        normal_path = os.path.join(comb_dir, "train/normal.aligned")
        simple_path = os.path.join(comb_dir, "train/simple.aligned")
        with open(normal_path, "w") as nfile, open(simple_path, "w") as sfile:
            for nsent, ssent in test:
                if do_tok:
                    nsent = " ".join(nltk.word_tokenize(nsent))
                    ssent = " ".join(nltk.word_tokenize(ssent))
                nfile.write(nsent + "\n")
                sfile.write(ssent + "\n")

        normal_path = os.path.join(comb_dir, "test/normal.aligned")
        simple_path = os.path.join(comb_dir, "test/simple.aligned")
        with open(normal_path, "w") as nfile, open(simple_path, "w") as sfile:
            for nsent, ssent in test:
                if do_tok:
                    nsent = " ".join(nltk.word_tokenize(nsent))
                    ssent = " ".join(nltk.word_tokenize(ssent))
                nfile.write(nsent + "\n")
                sfile.write(ssent + "\n")

    fn = os.path.join(split_dir, "vocab.pickle")
    with open(fn, "wb") as vfile:
        pickle.dump(vocab, vfile)


#split_newsela_sents_versions("/data/data/newsela_article_corpus_2016-01-29/newsela_data_share-20150302/newsela_articles_20150302.aligned.sents.txt",
#                   "/data/data/newsela")
split_newsela_sents_versions("/data/data/newsela_article_corpus_2016-01-29/newsela_data_share-20150302/newsela_articles_20150302.aligned.sents.txt",
                   "/data/data/newsela_tok", do_tok=True)


def load_wiki_sents(data_dir):
    """
    Load sentence pairs from wikipedia set.

    Parameters
    ----------
    data_dir : str
        Directory containing the files with
        aligned normal and simple sentences.
    """

    normal_path = os.path.join(data_dir, "normal.aligned")
    simple_path = os.path.join(data_dir, "simple.aligned")

    with open(normal_path) as nfile:
        lines = nfile.readlines()
        nsents = [l.split("\t")[2].strip() for l in lines]
    with open(simple_path) as sfile:
        lines = sfile.readlines()
        ssents = [l.split("\t")[2].strip() for l in lines]

    pairs = list(zip(nsents, ssents))
    return pairs


def load_newsela_sents(data_dir):
    """
    Load sentence pairs from newsela set.

    Parameters
    ----------
    data_dir : str
        Directory containing the files with
        aligned normal and simple sentences.
    """

    normal_path = os.path.join(data_dir, "normal.aligned")
    simple_path = os.path.join(data_dir, "simple.aligned")

    with open(normal_path) as nfile:
        lines = nfile.readlines()
        nsents = [l.strip() for l in lines]
    with open(simple_path) as sfile:
        lines = sfile.readlines()
        ssents = [l.strip() for l in lines]

    pairs = list(zip(nsents, ssents))
    return pairs


def save_test(in_path, out_path):
    """
    Save test sentences to one sentence per line
    in order to use with easse.
    """
    sents =  []
    with open(in_path) as in_file:
        for line in in_file:
            sent = line.split("\t")[2]
            sents.append(sent)
    sents = sents[:200]
    with open(out_path, "w") as out_file:
        for sent in sents:
            out_file.write(sent)

#save_test("/data/data/wiki/sent_aligned_split/test/simple.aligned",
#          "/data/data/wiki/sent_aligned_split/test/simple_sents_200.txt")
#save_test("/data/data/wiki/sent_aligned_split/test/normal.aligned",
#          "/data/data/wiki/sent_aligned_split/test/normal_sents_200.txt")
