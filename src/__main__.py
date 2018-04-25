
import argparse
import sys
import os
# This method will activate the argument parser and will have all the paramenters that will  be pased through the code
def get_args(args):
    parser = argparse.ArgumentParser(description='Blah Blah')
    parser.add_argument('Mandatory', help='Help is on the way', type=str)
    parser.add_argument('--optional', help='Its still on he way', default="wahtever",type=str)
    
    return parser.parse_args(args)



def main(args=None):
    print ("Cilia")
    if args is None:
        args = sys.argv[1:]
    args = get_args(args)
