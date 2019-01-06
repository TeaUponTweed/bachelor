import sys
import os
import random
import itertools
import time
from shutil import copyfile
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
    'ADRIANNE_JANE',
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
    constestant_map = {c : f'images/{c}.png' for c in constestants}

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
                    text = input("{} (a) or {} (b)? ({}/{}): ".format(a, b, i+1, len(comps)))
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


def get_sorted_ixs(itr):
    nodes, ixs = zip(*sorted(zip(itr, range(len(itr)))))
    return ixs, nodes


def score_sort(file):
    print("Maximizing number of satisfied pairs though CEM magic...")
    with open(file) as f:
        data = [l.strip().split(',') for l in f]
    if len(data[0]) == 4:
        winners, losers, top_or_bottom, times = zip(*data)
    else:
        winners, losers = zip(*data)
    first_better_than_second = set(zip(winners, losers))

    # setup CEM
    Sigma = np.eye(len(constestants))*10
    mu = np.zeros(len(constestants))
    N = int(50 * len(constestants)**2)

    def ranked_consistency(mu):
        ixs, nodes = get_sorted_ixs(mu)
        ix_map = dict(zip([constestants[ix] for ix in ixs], nodes))
        score = 0
        for (better, worse) in first_better_than_second:
            if ix_map[better] > ix_map[worse]:
                score += 1
        return -score

    x = CEMMinimizer(distribution=make_normal(mu, Sigma), infer_distribtion=infer_normal, nsamples=N, verbose=False).minimize(ranked_consistency)

    print('Matched', -ranked_consistency(x))
    ixs, _ = get_sorted_ixs(x)
    # print(ixs)
    for ix in reversed(ixs):
        print(constestants[ix])

def simple_sort(file):
    print("Applying pretty darn good heuristic...")

    with open(file) as f:
        data = [l.strip().split(',') for l in f]
    if len(data[0]) == 4:
        winners, losers, top_or_bottom, times = zip(*data)
        print(sum(tob == 'top' for tob in top_or_bottom)/len(top_or_bottom))
    else:
        winners, losers = zip(*data)

    first_better_than_second = set(zip(winners, losers))
    counts = Counter(winners)
    for constestant in constestants:
        if constestant not in counts:
            counts[constestant] = 0

    cs, order = zip(*sorted((c, k) for k, c in counts.items()))
    def score_exhaustive_order(order):
        score = 0
        for ab in itertools.combinations(order, 2):
            if ab in first_better_than_second:
                score += 1
        return score

    def gen_optimal():
        for c, suborder in itertools.groupby(zip(cs, order), lambda x: x[0]):
            _, suborder = zip(*suborder)
            suborder = max(itertools.permutations(suborder), key=score_exhaustive_order)
            yield from reversed(suborder)

    order = list(gen_optimal())

    ix_map = dict(zip(order, range(len(order))))

    score = 0
    for (better, worse) in first_better_than_second:
        if ix_map[better] > ix_map[worse]:
            score += 1
    print('Matched', score)
    for constestant in reversed(order):
        print(constestant)
    # # get number of ties
    # print(sum(c for c in Counter(counts.values()).values() if c > 1))

def test():
    # make fake data
    constestants, data = make_data()

    # setup CEM
    Sigma = np.eye(len(constestants))*10
    mu = np.zeros(len(constestants))
    N = int(500 * len(constestants)**2)

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
    outfile = 'michael.txt'
    generate_score(outfile)
    # plot_stats(outfile)
    # simple_sort(outfile)
    # score_sort(outfile)

