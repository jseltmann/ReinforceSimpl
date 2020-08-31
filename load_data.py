import os
import random
import nltk

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

save_test("/data/data/wiki/sent_aligned_split/test/simple.aligned",
          "/data/data/wiki/sent_aligned_split/test/simple_sents_200.txt")
save_test("/data/data/wiki/sent_aligned_split/test/normal.aligned",
          "/data/data/wiki/sent_aligned_split/test/normal_sents_200.txt")
