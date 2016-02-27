# -*- coding: utf-8 -*-

import sys
import warnings

import Gridsearcher

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data = sys.argv[1]
    # data = "a9a"
    Gridsearcher.gridsearch_and_save(data)
