import sys
import warnings

import ParameterTuning

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data = sys.argv[1]
    ParameterTuning.run(data)
