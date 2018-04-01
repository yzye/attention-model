# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:53:42 2018

@author: yuzhe
"""
import numpy as np
import tensorflow as tf
import jieba
import re
import itertools
from random import shuffle
import argparse
from collections import Counter


def input_doc(filename):
    '''
    eliminate na or 没有描述
    '''
    
    reviews = []
    r = "[\s+\.\!\/_,$%^*)(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    with open(filename, 'r', encoding = "utf-8") as f:
        for lines in f:        
            if re.sub(r,'',lines) not in ["na","没有描述"]:
                reviews.append([re.sub(r,'',x) for x in jieba.cut(lines) if len(re.sub(r,'',x))])
    return reviews

def get_labels(reviews_pos,reviews_neg):
    
    labels = np.array([1]*len(reviews_pos)+[0]*len(reviews_neg))
    labels = labels.reshape([-1,1])
    labels = np.hstack((1 - labels, labels))
    
    return labels

def reviews_encode(reviews):
    
    words = list(itertools.chain.from_iterable(reviews))
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    
    reviews_ints = []
    for each in reviews:
        reviews_ints.append([vocab_to_int[word] for word in each])
        
    return reviews_ints, vocab_to_int

def clean_reviews(reviews_ints,labels):
    '''
    eliminate vacant reviews and shuffle the data
    '''
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
    shuffle(non_zero_idx)
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    labels = np.array([labels[ii] for ii in non_zero_idx])
    
    return reviews_ints,labels

def padding(reviews_ints,seq_len):
    
    features = np.zeros((len(reviews_ints), seq_len), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]
    
    return features

def split_features(features,labels,split_frac):
    
    split_idx = int(len(features)*split_frac)

    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]

    return train_x, val_x, train_y, val_y

def single_cell(num_hidden, keep_prob):
    cell = tf.contrib.rnn.GRUCell(num_hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell

def my_attention(inputs, hidden_layer_size):
  
    X = tf.reshape(inputs, [-1, 2*num_hidden])
    Y = tf.layers.dense(X, hidden_layer_size, activation=tf.nn.relu)
    logits = tf.layers.dense(Y, 1, activation = None)
  
    logits = tf.reshape(logits, [-1, seq_len, 1])
    alphas = tf.nn.softmax(logits, axis=1)
    encoded_sentence = tf.reduce_sum(inputs * alphas, axis=1)

    return encoded_sentence, alphas

def get_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

def key_words(inputs, alphas):
    
    alphas.flatten()
    alphas.shape = (len(inputs),seq_len)
    
    index = [np.argmax(a) for a in alphas]
    int_words = [sen[i] for i,sen in zip(index,inputs)]
    key_words = [[key for key, value in vocab_to_int.items() if value == int_word][0] for int_word in int_words]
    
    return key_words

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pos_data", help="Positive data for training",
                           type=str, default=None, required=False)
    argparser.add_argument("--neg_data", help="Negative data for training",
                           type=str, default=None, required=False)
    
    argparser.add_argument("--new_reviews", help="New reviews for prediction",
                           type=str, default=None, required=False)
    
    args = argparser.parse_args()
    
    reviews_pos = input_doc(args.pos_data)
    reviews_neg = input_doc(args.neg_data)

    reviews = reviews_pos + reviews_neg
    labels = get_labels(reviews_pos,reviews_neg)
    
    reviews_ints,vocab_to_int = reviews_encode(reviews)
    reviews_ints,labels = clean_reviews(reviews_ints,labels)
    
    seq_len = 80
    features = padding(reviews_ints,seq_len)

    split_frac = 0.8
    train_x, val_x, train_y, val_y = split_features(features,labels,split_frac)
    
    n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1
    
    # Training Parameters
    learning_rate = 0.001
    epochs = 10
    batch_size = 64
    display_step = 200

    # Network Parameters
    embed_size = 300
    num_hidden = 50 # hidden layer num of features
    num_classes = 2
    att_hidden = 32
    
    # tf Graph input
    inputs_ = tf.placeholder(tf.int32, [None, seq_len], name="input")
    labels_ = tf.placeholder(tf.int32, [None, num_classes], name="labels")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    embedding = tf.Variable(tf.random_uniform([n_words, embed_size]))
    embed = tf.nn.embedding_lookup(embedding, inputs_)

    rnn_fw_cell = single_cell(num_hidden, keep_prob)
    rnn_bw_cell = single_cell(num_hidden, keep_prob)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, rnn_bw_cell, embed, dtype=tf.float32)
    outputs = tf.concat(outputs, axis = 2)

    encoded, alphas = my_attention(outputs, att_hidden)

    logits = tf.layers.dense(encoded, 2, activation=None)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        step = 0
        for epoch in range(epochs):
            for batch_x, batch_y in get_batches(train_x[:64000], train_y[:64000], batch_size):
                step += 1
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={inputs_: batch_x, labels_: batch_y, keep_prob: 0.8})
                if step % display_step == 0 or step == 1:
                    loss, acc, a = sess.run([loss_op, accuracy, alphas], feed_dict={inputs_: batch_x,
                                                                         labels_: batch_y,
                                                                         keep_prob: 0.8})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc), ", Testing Accuracy=", \
                          sess.run(accuracy, feed_dict={inputs_: val_x[:5000,:], labels_: val_y[:5000,:], keep_prob:1}))

        print("Optimization Finished!")
        saver.save(sess, "checkpoints/sentiment.ckpt")

    new_reviews = input_doc(args.new_reviews)
    
    new_reviews_ints = []
    for sen in new_reviews:
        new_reviews_ints.append([vocab_to_int[word] for word in sen if word in vocab_to_int])
    
    features = padding(new_reviews_ints,seq_len)


    test_acc = []
    with tf.Session() as sess:
        
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        feed = {inputs_: features,
                    labels_: [[0,1]],
                    keep_prob: 1}
        pred,alphas = sess.run([prediction,alphas], feed_dict=feed)
        key_word = key_words(features, alphas)
        
        with open("prediction.txt",'w') as output:
            output.write("prediction"+'\t'+"key feature word"+'\n')
            for p, k in zip(pred,key_word):
                label_pred = ("negative","positive")[np.argmax(p)]
                output.write('\n'+label_pred+'\t'+k)



