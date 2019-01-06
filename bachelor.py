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


if __name__ == '__main__':
    generate_score(sys.argv[1])
