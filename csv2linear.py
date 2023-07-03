#!/usr/bin/env python3

import argparse
import sys

def csv2linear(line:str, sep:str=',', excludeBlank=True) -> list[str]:

    rtn = []
    if sep in line:
       rtn = line.split(sep)
    else:
       rtn = [line]

    if excludeBlank:
       rtn = [v  for v in rtn if v != '']
    return rtn


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',        type=str,      default='/dev/stdin',                     help='path of input')           # input file path
    parser.add_argument('-o', '--output',       type=str,      default='/dev/stdout',                    help='path of output')          # output file path
    parser.add_argument('-s', '--separator',    type=str,      default=',',                              help='separator of value')      # value separator
    parser.add_argument('-e', '--excludeblank', type=bool,     default=True,                             help='exclude blank value')     #
    args = parser.parse_args()
    print(args, file=sys.stderr)

with open(args.input, encoding='utf-8') as fp:
     content = fp.read().splitlines()

rtn = []
for l in content:
    rtn.extend ( csv2linear(l, sep=args.separator, excludeBlank=args.excludeblank))

with open(args.output, encoding='utf-8', mode='w') as out:
    for v in rtn:
        print(v, file=out)
