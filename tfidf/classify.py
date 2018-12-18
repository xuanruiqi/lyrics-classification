import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

vectorizer = TfidfVectorizer()
clf = MultinomialNB()

def clean_line(line):
    return line.replace('\n', '').replace('\u3000', '') \
                                 .replace('?', '') \
                                 .replace('!', '') \
                                 .replace('(', '').replace(')', '')

def load_data_files(tag, dirname):
    """
    Load the data files and do some cleanup and pre-processing
    """
    filenames = os.listdir(dirname)

    data = []
    
    for fn in filenames:
        if fn.endswith("_processed.txt"):
            with open(os.path.join(dirname, fn), "r") as f:
                lines = []

                for line in f:
                    line = clean_line(line)
                    # clean_line_array(line_words)
                    lines.append(line)

                train = np.random.randint(0, 10)

                lines_str = ' '.join(lines)

                # print(lines_str)
                
                datum = {'category' : tag,
                         'lines' : lines_str,
                         'train?' : train < 9}
                data.append(datum)

    return data


def separate_training_testing(data):
    train, test = [], []

    for datum in data:
        if datum['train?']:
            train.append(datum)
        else:
            test.append(datum)

    return train, test


def separate_doc_tag(data):
    docs, tags = [], []
    
    for datum in data:
        docs.append(datum['lines'])
        tags.append(datum['category'])

    return docs, tags


def vectorize_docs(docs):
    X = vectorizer.fit_transform(docs)
    return X


def train_model(docs, tags):
    X = vectorize_docs(docs)
    y = np.array(tags)
    global clf
    clf = clf.fit(X, y)


def predict(doc):
    X_test = vectorizer.transform([doc])
    return clf.predict(X_test)


def benchmark(test_docs, test_tags):
    X_test = vectorizer.transform(test_docs)
    y_test = np.array(test_tags)

    pred = clf.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)

    return score

if __name__ == '__main__':
    dataset = {'spring': 'spring', 'summer': 'summer', 'autumn': 'autumn',
               'winter': 'winter'}

    data = []
    
    for tag, dirname in enumerate(dataset):
        data += load_data_files(tag, dirname)

    train, test = separate_training_testing(data)

    print('Training set: {}, testing set: {}'.format(len(train), len(test)))

    train_doc, train_tag = separate_doc_tag(train)
    test_doc, test_tag = separate_doc_tag(test)

    train_model(train_doc, train_tag)

    score = benchmark(test_doc, test_tag)
    print('Accuracy score: {}'.format(score))
