#! /usr/bin/env python3

import sys
import os
import re
from pyknp import Juman

jpp = Juman()

def process_line(line):
    base_words = []
    try:
        result = jpp.analysis(line)
    except Exception:
        return []
    for mrph in result.mrph_list():
        if mrph.hinsi != '助詞' and \
           mrph.hinsi != '助動詞' and \
           mrph.hinsi != '記号' and \
           not '非自立' in mrph.bunrui and \
           not '接尾' in mrph.bunrui:
            base_words.append(mrph.genkei)

    return base_words


def process_file(filename):
    processed_lines = []
    
    with open(filename, 'r') as f:
        for line in f:
            processed = process_line(line)
            new_line = ' '.join(processed)
            new_line = re.sub(r"\(.*\)", "", new_line)
            new_line = new_line.replace('(', '').replace(')', '').replace(',', '')
            processed_lines.append(new_line)
            
    return '\n'.join(processed_lines)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Provide a file to parse by line')

    filename = sys.argv[1]
    processed_lines = []
    
    with open(filename, 'r') as f:
        for line in f:
            processed = process_line(line)
            processed_lines.append(' '.join(processed))

    save_filename = "{}_processed.txt".format(os.path.splitext(filename)[0])

    with open(save_filename, 'w+') as f:
        for line in processed_lines:
            f.write("{}\n".format(line))
