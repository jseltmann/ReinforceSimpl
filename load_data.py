import os
import random

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

    with open(normal_path) as nfile:
        normal_lines = nfile.readlines()
    with open(simple_path) as sfile:
        simple_lines = sfile.readlines()

    zipped = list(zip(normal_lines, simple_lines))
    random.shuffle(zipped)
    
    cutoff = int(0.9 * len(zipped))
    train = zipped[:cutoff]
    test = zipped[cutoff:]

    normal_path = os.path.join(split_dir, "normal.train.aligned")
    simple_path = os.path.join(split_dir, "simple.train.aligned")
    with open(normal_path, "w") as nfile, open(simple_path, "w") as sfile:
        for nline, sline in train:
            nfile.write(nline)
            sfile.write(sline)

    normal_path = os.path.join(split_dir, "normal.test.aligned")
    simple_path = os.path.join(split_dir, "simple.test.aligned")
    with open(normal_path, "w") as nfile, open(simple_path, "w") as sfile:
        for nline, sline in test:
            nfile.write(nline)
            sfile.write(sline)

#split_wiki_sent("/data/sentence-aligned.v2",
#                "/data/wiki/sent_aligned_split")
