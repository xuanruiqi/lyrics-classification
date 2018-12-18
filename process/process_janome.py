#! /usr/bin/env python3

import sys
import os
import re
from janome.tokenizer import Tokenizer

t = Tokenizer()

def process_line(line):
    base_words = []
    
    for token in t.tokenize(line):
        pos = token.part_of_speech
        if not '助詞' in pos and \
           not '記号' in pos and \
           not '接尾' in pos and \
           not '非自立' in pos and \
           not '助動詞' in pos:
            base_words.append(token.base_form)

    return base_words


def process_file(filename):
    processed_lines = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = re.sub(r"\(.*\)", "", line)
            processed = process_line(line)
            new_line = ' '.join(processed)
            new_line = new_line.replace('(', '').replace(')', '').replace(',', '')
            processed_lines.append(new_line)

    return '\n'.join(processed_lines)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Provide a file to parse by line')
        sys.exit(1)

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
