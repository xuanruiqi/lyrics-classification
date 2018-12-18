# Adapted from: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/train.py
# and: https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn/blob/master/train.py

import os
import sys
import json
import time
import shutil
import pickle
import logging
import process_data
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from text_cnn import TextCNN
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
logging.getLogger().setLevel(logging.INFO)

FLAGS = tf.flags.FLAGS

def load_word_embeddings(vocab):
    kv = KeyedVectors.load('ja-wv.wv', mmap='r')

    for word in vocab:
        if word not in kv:
            kv.add([word], [np.random.uniform(-0.25,0.25,50)])

    return kv


def return_word_embedding(kv, word):
    if word in kv:
        return kv[word]
    else:
        return np.random.uniform(-0.25,0.25,50)


def train_cnn_rnn(dataset):
    x_, y_, vocabulary, vocabulary_inv, labels = process_data.process_data(dataset)

    training_config = sys.argv[1]
    params = json.loads(open(training_config).read())
    
    # Assign a 300 dimension vector to each word
    # word_embeddings = load_word_embeddings(vocabulary)
    kv = KeyedVectors.load('ja-wv.wv', mmap='r')
    print("Loaded word embeddings!")
    
    embedding_mat = [return_word_embedding(kv, word) for word in list(vocabulary_inv)]
    embedding_mat = np.array(embedding_mat, dtype = np.float32)
    print("Generated embedding matrices")
    
    # Split the original dataset into train set and test set
    x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1)

    # Split the train set into train set and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)
    
    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))
    
    # Create a directory, everything related to the training will be saved in this directory
    timestamp = str(int(time.time()))
    trained_dir = './trained_results_' + timestamp + '/'
    if os.path.exists(trained_dir):
        print('remaking trained dir...')
        shutil.rmtree(trained_dir)
    os.makedirs(trained_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                embedding_mat=embedding_mat,
                sequence_length=x_train.shape[1],
                vocab_size=len(list(vocabulary_inv)),
                num_classes = y_train.shape[1],
                batch_size = params['batch_size'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters = params['num_filters'],
                embedding_size = params['embedding_dim'],
                l2_reg_lambda = params['l2_reg_lambda'])
        
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            with open(trained_dir + 'words_index.json', 'w+') as outfile:
                json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: params['dropout_keep_prob']
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = process_data.batch_iter(
                list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % params['evaluate_every'] == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % params['evaluate_every'] == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    

if __name__ == '__main__':
    # python3 train.py ./training_config.json
    dataset = {'spring': 'spring', 'summer': 'summer', 'autumn': 'autumn',
               'winter': 'winter'}
    train_cnn_rnn(dataset)
