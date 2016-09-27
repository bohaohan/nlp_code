import tensorflow as tf
import numpy as np
from ConvolutionalBatchNormalizer import ConvolutionalBatchNormalizer as BN

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    1D one hot
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        num_filters1 = 8
        num_filters2 = 16
        num_filters3 = num_filters
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        print self.embedded_chars_expanded.get_shape()
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer size1
                filter_shape = [filter_size, 101, 1, num_filters1]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters1]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply BatchNormalization

                ewma = tf.train.ExponentialMovingAverage(decay=0.99)
                bn = BN(num_filters1, 0.001, ewma, True)
                update_assignments = bn.get_assigner()
                bn1 = bn.normalize(conv, train=True)


                relu1 = tf.nn.relu(tf.nn.bias_add(bn1, b), name="relu")

                # Convolution layer 2

                filter_shape2 = [2, 101, num_filters1, num_filters2]
                W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W1")
                b2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]), name="b2")
                conv = tf.nn.conv2d(
                    relu1,
                    W2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply BatchNormalization

                ewma2 = tf.train.ExponentialMovingAverage(decay=0.99)
                bn2 = BN(num_filters2, 0.001, ewma2, True)
                update_assignments = bn2.get_assigner()
                bn22 = bn2.normalize(conv, train=True)


                relu2 = tf.nn.relu(tf.nn.bias_add(bn22, b2), name="relu")


                print filter_size, "before pooling", relu2.get_shape()
                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     relu1,
                #     ksize=[1, 2, 1, 1],
                #     strides=[1, 2, 1, 1],
                #     padding='VALID',
                #     name="pool")

                # Convolution pooling

                filter_shape_cm = [2, 2, num_filters2, num_filters2]
                W_cm = tf.Variable(tf.truncated_normal(filter_shape_cm, stddev=0.1), name="W1")
                b_cm = tf.Variable(tf.constant(0.1, shape=[num_filters2]), name="b2")
                conv_cm = tf.nn.conv2d(
                    relu2,
                    W_cm,
                    strides=[1, 2, 2, 1],
                    padding="VALID",
                    name="conv")

                # Apply BatchNormalization

                ewma_cm = tf.train.ExponentialMovingAverage(decay=0.99)
                bn_cm = BN(num_filters2, 0.001, ewma_cm, True)
                update_assignments = bn_cm.get_assigner()
                bn_cm_o = bn_cm.normalize(conv_cm, train=True)

                pooled = tf.nn.relu(tf.nn.bias_add(bn_cm_o, b_cm), name="relu")

                print filter_size, "pooling1", pooled.get_shape()



                filter_shape3 = [1, 27, num_filters2, num_filters3]
                W3 = tf.Variable(tf.truncated_normal(filter_shape3, stddev=0.1), name="W3")
                b3 = tf.Variable(tf.constant(0.1, shape=[num_filters3]), name="b3")
                conv3 = tf.nn.conv2d(
                    pooled,
                    W3,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv3")


                # Apply BatchNormalization
                ewma3 = tf.train.ExponentialMovingAverage(decay=0.99)
                bn_3 = BN(num_filters3, 0.001, ewma3, True)
                update_assignments = bn_3.get_assigner()
                bn3 = bn_3.normalize(conv3, train=True)


                # Apply nonlinearity
                relu3 = tf.nn.relu(tf.nn.bias_add(bn3, b3), name="relu3")

                # Maxpooling over the outputs

                # Convolution pooling
                w = 2
                # if filter_size == 3:
                #     w = 3

                # pooled3 = tf.nn.max_pool(
                #     relu3,
                #     # ksize=[1, 2, 15, 1],
                #     ksize=[1, w, 15, 1],
                #     strides=[1, 2, 1, 1],
                #     padding='VALID',
                #     name="pool3")

                # pooled3 = tf.nn.top_k(relu3, 10, sorted=False, name="k-max-pooling")
                filter_shape_cm2 = [w, 2, num_filters3, num_filters3]
                W_cm2 = tf.Variable(tf.truncated_normal(filter_shape_cm2, stddev=0.1), name="W1")
                b_cm2 = tf.Variable(tf.constant(0.1, shape=[num_filters3]), name="b2")
                conv_cm2 = tf.nn.conv2d(
                    relu3,
                    W_cm2,
                    strides=[1, 2, 2, 1],
                    padding="VALID",
                    name="conv")

                # Apply BatchNormalization

                ewma_cm2 = tf.train.ExponentialMovingAverage(decay=0.99)
                bn_cm2 = BN(num_filters3, 0.001, ewma_cm2, True)
                update_assignments = bn_cm2.get_assigner()
                bn_cm_o2 = bn_cm2.normalize(conv_cm2, train=True)

                pooled3 = tf.nn.relu(tf.nn.bias_add(bn_cm_o2, b_cm2), name="relu")

                pooled_outputs.append(pooled3)
        print pooled_outputs[0].get_shape()
        print pooled_outputs[1].get_shape()
        print pooled_outputs[2].get_shape()
        # Combine all the pooled features
        sum = 0
        for i in pooled_outputs:
            sum += i.get_shape()[1].value
        # Combine all the pooled features
        num_filters_total = num_filters * sum
        self.h_pool = tf.concat(1, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print "before drop out", self.h_pool_flat.get_shape()
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            print "after drop out", self.h_drop.get_shape()

        # with tf.name_scope("fullyConnected"):
        #     # Fully connected layer
        #     W = tf.Variable(tf.truncated_normal([num_filters_total, 256], stddev=0.1), name="W")
        #     b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
        #     dense1 = tf.reshape(self.h_drop, [-1, W.get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
        #     dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, W), b)) # Relu activation
        #     dense1 = tf.nn.dropout(dense1, self.dropout_keep_prob) # Apply Dropout

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
