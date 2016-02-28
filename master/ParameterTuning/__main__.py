# -*- coding: utf-8 -*-


import os
import sys
import warnings
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import Gridsearcher

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data = sys.argv[1]
    # data = "a9a"
    Gridsearcher.gridsearch_and_save(data)
