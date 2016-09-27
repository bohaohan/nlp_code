import tensorflow as tf
import numpy as np
from ConvolutionalBatchNormalizer import ConvolutionalBatchNormalizer as BN

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    345 64
    """
    def __init__(self, sequence_length, num_classes, height,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, batch_size=64):

        self.input_x = tf.placeholder(tf.float32, [None, height, embedding_size, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.input_x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply BatchNormalization

                ewma = tf.train.ExponentialMovingAverage(decay=0.99)
                bn = BN(num_filters, 0.001, ewma, True)
                update_assignments = bn.get_assigner()
                bn1 = bn.normalize(conv, train=True)

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(bn1, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                     h,
                     ksize=[1, sequence_length - filter_size + 1, 1, 1],
                     strides=[1, 1, 1, 1],
                     padding='VALID',
                     name="pool")

                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        print "before drop out", self.h_pool_flat.get_shape()

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            print "after drop out", self.h_drop.get_shape()

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
