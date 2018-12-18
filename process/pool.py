import os
import sys

import process_janome
import process_jumanpp
from multiprocessing import Pool


def get_files(dirname):
    files = os.listdir(os.path.abspath(dirname))
    ret = []
    for f in files:
        if f.endswith('.txt'):
            ret.append(f)
    return ret


def write_processed(dirname, filename, process_func):    
    result = process_func(os.path.join(dirname, filename))
    basename = os.path.splitext(filename)[0]
    write_path = os.path.join(dirname, "{}_processed.txt".format(basename))

    print("Writing to: {}".format(write_path))
    
    with open(write_path, 'w+') as f:
        f.write(result)

    return filename


def signal(filename):
    print("Done processing: {}".format(filename))


def main(dirname, tokenizer):
    print("Working on directory: {}".format(os.path.abspath(dirname)))
    
    process_func = None

    if tokenizer == 'janome':
        process_func = process_janome.process_file
    elif tokenizer == 'jumanpp':
        process_func = process_jumanpp.process_file
    else:
        raise Exception('Invalid tokenizer!')

    print("Using tokenizer: {}".format(tokenizer))
    
    filenames = get_files(dirname)
    
    with Pool(processes=4) as pool:
        for filename in filenames:
            print("Processing: {}".format(filename))
            pool.apply(write_processed, (dirname, filename, process_func))


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Provide a directory name and a tokenizer name")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
