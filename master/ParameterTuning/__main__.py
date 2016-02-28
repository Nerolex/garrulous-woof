# -*- coding: utf-8 -*-

import sys
import warnings

import Gridsearcher

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    data = sys.argv[1]
    # data = "a9a"
    Gridsearcher.gridsearch_and_save(data)
