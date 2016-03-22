# -*- coding: utf-8 -*-


import os
import sys
import warnings
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import Gridsearcher

'''
Main module for independet use of the Parameter Tuning. Takes a list of data strings as argument and automatically searches in /data/ for the corresponding dataset.
Performs a gridsearch with the module Gridsearcher and saves the results to output/NAME_OF_DATA-params.csv
'''

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data = ["cod-rna"]
    # data = sys.argv
    # data.pop(0)

    for datastring in data:
        # try:
        Gridsearcher.gridsearch_and_save(datastring)
        # except Exception:
        #continue
