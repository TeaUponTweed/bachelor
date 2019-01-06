import sys
import os
import random
import itertools
import time
from stat import S_ISREG, ST_CTIME, ST_MODE
from collections import Counter

import PIL

import numpy as np
import matplotlib.pyplot as plt

from cem import CEMMinimizer, make_normal, infer_normal

constestants = [
    'ALEX B.',
    'ALEX D.',
    'ANGELIQUE',
    'ANNIE',
    'BRI',
    'CAELYNN',
    'CAITLIN',
    'CASSIE',
    'CATHERINE',
    'COURTNEY',
    'DEMI',
    'DEVIN',
    'ELYSE',
    'ERIKA',
    'ERIN',
    'HANNAH B.',
    'HANNAH G.',
    'HEATHER',
    'ADRIANNE "JANE"',
    'KATIE',
    'KIRPA',
    'LAURA',
    'NICOLE',
    'NINA',
    'ONYEKA',
    'REVIAN',
    'SYDNEY',
    'TAHZJUAN',
    'TAYSHIA',
    'TRACY',
]

def generate_score(outfile):

    candidate_dir = '/Users/MichaelMason/Desktop/bachelor_candidates/'
    # get all entries in the directory w/ stats
    entries = (os.path.join(candidate_dir, fn) for fn in os.listdir(candidate_dir))
    entries = ((os.stat(path), path) for path in entries)

    # leave only regular files, insert creation date
    entries = ((stat[ST_CTIME], path)
               for stat, path in entries if S_ISREG(stat[ST_MODE]))
    pictures = [os.path.join(candidate_dir, f) for _, f in sorted(entries) if 'png' in f]
    # images = [PIL.Image.open(os.path.join(candidate_dir, f)) for f in pictures if 'png' in f]
    constestant_map = dict(zip(constestants, pictures))
    # for constestant, img_file in constestant_map.items():
    #     with PIL.Image.open(img_file) as img:
    #         img.show()
    #         time.sleep(1.0)

    with open(outfile, 'w') as out:
        comps = list(itertools.combinations(constestants, 2))
        random.shuffle(comps)
        for ix in range(len(comps)):
            comps[ix] = list(comps[ix])
            random.shuffle(comps[ix])

        for i, (a, b) in enumerate(comps):
            if a != b:
                f, (ax1, ax2) = plt.subplots(2)
                # f.set_size_inches(18.5, 10.5)
                ax1.imshow(PIL.Image.open(constestant_map[a]))
                ax2.imshow(PIL.Image.open(constestant_map[b]))
                plt.tight_layout()
                plt.ion()
                plt.show()
                start =time.time()
                while True:
                    text = input("{} (a) or {} (b)? ({}/{})".format(a, b, i+1, len(comps)))
                    end = time.time()
                    if text == "a":
                        print(f"{a},{b},top,{end-start}", file=out)
                        break
                    elif text == "b":
                        print(f"{b},{a},bottom,{end-start}", file=out)
                        break
                    elif text == "q":
                        return
                    else:
                        print(f'unrecognized input {text}')
                plt.close()
            out.flush()

def make_data():
    constestants = [
        'Al',
        'Tyrell',
        'Bud',
        'Errol',
        'Kip',
        'Merrill',
        'Archie',
        'Armand',
        'Marshall',
        'Sebastian',
        'Frank',
        'Dustin',
        'Bill',
        'Vito',
        'Dylan',
        'Lyle',
        'Francis',
        'Fredrick',
        'Jimmy',
        'Britt',
    ]
    ncontestents = len(constestants)
    first_better_than_second = set()

    for i in range(ncontestents):
        for j in range(i + 1, ncontestents):
            if random.random() < 0.3:
                first_better_than_second.add((i,j))

    return constestants, first_better_than_second

def plot_stats(*files):
    assert 0 < len(files) < 3
    f, ax = plt.subplots(1)
    for file, width in zip(files, (0.4, -0.4)):
        with open(file) as f:
            data = [l.split(',') for l in f]
        if len(data[0]) == 4:
            winners, losers, top_or_bottom, times = zip(*data)
            print(sum(tob == 'top' for tob in top_or_bottom)/len(top_or_bottom))
        else:
            winners, losers = zip(*data)

        counts = Counter(winners)
        y = []
        for ix, constestant in enumerate(constestants):
            y.append(counts[constestant])


        x = list(range(len(y)))
        ax.bar(x, y, label=file, align='edge', width=width)
        ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels(constestants, rotation=90    )

    plt.show()

def test():
    # make fake data
    constestants, data = make_data()


    # setup CEM
    Sigma = np.eye(len(constestants))*10
    mu = np.zeros(len(constestants))
    N = int(500 * len(constestants)**2)
    def get_sorted_ixs(itr):
        nodes, ixs = zip(*sorted(zip(itr, range(len(itr)))))
        return ixs, nodes

    # define objective
    def ranked_consistency(mu):
        ixs, nodes = get_sorted_ixs(mu)
        ix_map = dict(zip(ixs, nodes))
        score = 0
        for (better, worse) in data:
            if ix_map[better] > ix_map[worse]:
                score += 1
        return -score

    x = CEMMinimizer(distribution=make_normal(mu, Sigma), infer_distribtion=infer_normal, nsamples=N, verbose=True).minimize(ranked_consistency)

    print('min distance ', ranked_consistency(x))
    ixs, _ = get_sorted_ixs(x)
    print(ixs)
    for ix in ixs:
        print(constestants[ix])


if __name__ == '__main__':
    # test()
    # generate_score('hanna.txt')
    plot_stats(*sys.argv[1:])

