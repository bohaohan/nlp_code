
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, height,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, batch_size=64):

        # Placeholders for input, output and dropout

        self.input_x = tf.placeholder(tf.float32, [None, height, embedding_size, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # Create a convolution + maxpool layer for each filter size
        num_filters1 = 16
        num_filters2 = 16
        num_filters3 = num_filters
        print "input size", height, embedding_size, 1
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, 151, 1, num_filters1]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters1]), name="b")
                conv = tf.nn.conv2d(
                    self.input_x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                print filter_size, "before pooling", h.get_shape()
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 2, 1, 1],
                    strides=[1, 2, 1, 1],
                    padding='VALID',
                    name="pool")
                print filter_size, "pooling1", pooled.get_shape()

                #  # third convelution-pooling layer
                filter_shape3 = [filter_size, 150, num_filters2, num_filters3]
                W3 = tf.Variable(tf.truncated_normal(filter_shape3, stddev=0.1), name="W3")
                b3 = tf.Variable(tf.constant(0.1, shape=[num_filters3]), name="b3")
                conv3 = tf.nn.conv2d(
                    pooled,
                    W3,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv3")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
                # Maxpooling over the outputs
                pooled3 = tf.nn.max_pool(
                    h,
                    ksize=[1, (((sequence_length - filter_size + 1)/2) - filter_size + 1), 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool3")

                pooled_outputs.append(pooled3)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # print "before drop out", self.h_pool_flat.get_shape()
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

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

