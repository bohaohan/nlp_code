
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
        # self.input_x = tf.placeholder(tf.float32, [None, sequence_length], name="input_x")
        # self.input_x = tf.placeholder(tf.float32, [None, 30000], name="input_x")
        self.input_x = tf.placeholder(tf.float32, [None, height, embedding_size, 1], name="input_x")
        # [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        sentence = self
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     W = tf.Variable(
        #         tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        #         name="W")
        #     self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        #     self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

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

                # # second convelution-pooling layer
                # filter_shape2 = [filter_size, 101, num_filters1, num_filters2]
                # W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W2")
                # b2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]), name="b2")
                # conv2 = tf.nn.conv2d(
                #     pooled,
                #     W2,
                #     strides=[1, 1, 1, 1],
                #     padding="VALID",
                #     name="conv2")
                # # Apply nonlinearity
                # h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
                # print filter_size, "before pooling2", h2.get_shape()
                # # Maxpooling over the outputs
                # pooled2 = tf.nn.avg_pool(
                #     h2,
                #     ksize=[1, 2, 1, 1],
                #     strides=[1, 2, 1, 1],
                #     padding='VALID',
                #     name="pool2")
                # print filter_size, "pooling2", pooled2.get_shape()
                #  # third convelution-pooling layer
                # filter_shape3 = [filter_size, 100, num_filters2, num_filters3]
                # W3 = tf.Variable(tf.truncated_normal(filter_shape3, stddev=0.1), name="W3")
                # b3 = tf.Variable(tf.constant(0.1, shape=[num_filters3]), name="b3")
                # conv3 = tf.nn.conv2d(
                #     pooled2,
                #     W3,
                #     strides=[1, 1, 1, 1],
                #     padding="VALID",
                #     name="conv3")
                # # Apply nonlinearity
                # h = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
                # # Maxpooling over the outputs
                # pooled3 = tf.nn.avg_pool(
                #     h,
                #     ksize=[1, (((sequence_length - filter_size + 1)/2 - filter_size + 1)/2 - filter_size + 1), 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool3")
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
        print pooled_outputs[0].get_shape()
        #print pooled_outputs[1].get_shape()
        #print pooled_outputs[2].get_shape()
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # print "before drop out", self.h_pool_flat.get_shape()
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        # print "after drop out", self.h_drop.get_shape()
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

