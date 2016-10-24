from __future__ import with_statement
from __future__ import print_function

from argparse import ArgumentParser

import sys
import os.path

# Parse options and arguments
parser = ArgumentParser()
parser.add_argument('data_folder', type=str,
                    help='the name of the folder containing the text files')
parser.add_argument('--verbose', action='store_true',
                    help='if set, output will be more verbose')
parser.add_argument("--chi2_select",
                    action="store", type=int, dest="select_chi2",
                    help="select some number of features using a chi-squared test")
parser.add_argument('--test',
                    action='store', type=int, dest='test_fraction',
                    help='if set, only a fraction of the data will be trained on and no cross-validation will be used')
parser.add_argument("--dev",
                    action="store_true",
                    help="if set, accuracy will be measured against a 30 percent dev set. Cannot be used in tandem with --cv_range.")
parser.add_argument("--predict",
                    action="store_true",
                    help="If set, predictions will be made for the unknown test data")

args = parser.parse_args()

if not os.path.exists(args.data_folder):
    parser.error('The folder %s does not exist' % args.data_folder)
    sys.exit(1)

print(__doc__)
print()
parser.print_help()
print()



if __name__ == '__main__':
