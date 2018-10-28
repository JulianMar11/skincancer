import tensorflow as tf
import tensorflow.contrib.layers as layers
import models.cnn_models as fe
import models.metrics as me
import models.util as util
# import numpy as np
# import cv2
import os
from os.path import join


class AbstractModel(object):
    def __init__(self, path, name, rewrite=False, learning_rate=1e-4):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.model = {}
        self.global_step = 0
        self.directory = os.path.join(path, name)
        self.save_dir = join(self.directory, 'net/')
        self.starter_learning_rate = learning_rate

        if rewrite:
            util.clear_dir(self.directory)
        else:
            util.mk_dirs(self.directory)

    def inference(self):
        pass

    def loss(self, output, labels):
        pass

    def input_setup(self):
        pass

    def merge(self):
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.directory + '/logs/train', self.sess.graph)
        test_writer = tf.summary.FileWriter(self.directory + '/logs/test')
        writer = [train_writer, test_writer]
        self.model['merged'] = merged
        self.model['writer'] = writer
        return merged, writer

    def training_setup(self, loss):
        """Sets up the models Ops.
        Creates a summarizer for various parameters in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
        Returns:
        train_op: The Op for models.
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        '''
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate,  # Base learning rate.
                                                   self.global_step * batch_size,  # Current index into the dataset.
                                                   50000,  # Decay step.
                                                   0.95,  # Decay rate.
                                                   staircase=True)
        '''
        # Summaries
        tf.summary.scalar('learning_rate', self.starter_learning_rate)
        util.add_summaries()
        with tf.name_scope('optimizer'):
            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.AdamOptimizer(self.starter_learning_rate)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single models step.
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        self.model["optimizer"] = train_op
        return train_op

    def init_model(self):
        # Initialize weight and biases
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        saver = tf.train.Saver(name='saver')
        self.model["saver"] = saver
        print("Finished building model")
        print("Total parameters: " + str(util.count_parameters()))

    def save(self, write_meta_graph=True):
        print("SAVING MODEL")
        name = self.save_dir + 'net'
        self.model["saver"].save(self.sess, name, global_step=self.global_step, write_meta_graph=write_meta_graph)

    def get_model_from_graph(self, parameters):
        for name in parameters:
            self.model[name] = self.graph.get_tensor_by_name(name + ":0")

    def summary(self, x_batch, y_batch, mode="train"):
        if mode == "train":
            j = 0
        else:
            j = 1
        summary_train, _, i = self.sess.run([self.model["merged"], self.model["loss"], self.global_step], feed_dict={self.model["x"]: x_batch, self.model["y"]: y_batch})
        self.model["writer"][j].add_summary(summary_train, i)

    def evaluate(self, x_batch, y_batch, mode="train"):
        pass

    def predict(self, x_batch):
        return self.sess.run(self.model["y"], feed_dict={self.model["x"]: x_batch})


class ClassificationModel(AbstractModel):
    def __init__(self, path, input_shape=(150, 220, 1), classnames=None, extractor=fe.CNN, training=False, learning_rate=1e-4, dropout=0.0, rewrite=False):
        self.name = "Classification_" + extractor.__name__ + '_d' + str(dropout) + '_l' + str(learning_rate)
        super(ClassificationModel, self).__init__(path, self.name, rewrite, learning_rate)
        self.input_shape = input_shape
        self.training = training
        self.extractor = extractor
        self.dropout = dropout
        self.classnames = classnames
        self.classes = len(classnames)
        self.model_parameters = ['x', 'y', 'loss', 'accuracy', 'global_step', 'predictions', 'predictions_op', 'correct_prediction_op', 'merged', 'optimizer']

    def input_setup(self):
        x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape[0], self.input_shape[1], self.input_shape[2]], name="x")
        y = tf.placeholder(tf.int64, [None], name="y")
        self.model['x'] = x
        self.model['y'] = y
        return x, y

    def loss(self, output, labels):
        predictions = tf.nn.softmax(output)
        # Define the accuracy calculation
        with tf.name_scope('Accuracy'):
            predictions_op = tf.argmax(predictions, 1)
            correct_prediction_op = tf.equal(predictions_op, tf.to_int64(labels))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction_op, tf.float32))
            tf.summary.scalar('Accuracy', accuracy_op)
            self.model["predictions"] = predictions
            self.model["predictions_op"] = predictions_op
            self.model["correct_prediction_op"] = correct_prediction_op
            self.model["accuracy"] = accuracy_op

        loss = me.sparse_xentropy_loss(output, labels)
        tf.summary.scalar('loss', loss)
        self.model["loss"] = loss
        return loss

    def inference(self):
        print("BUILDING GRAPH")
        keep_prob = 1.0 - self.dropout
        normalizer_params = {'is_training': self.training, 'updates_collections': None, 'decay': 0.99}
        initializer = layers.variance_scaling_initializer()
        activation_fn = tf.nn.relu
        normalizer_fn = layers.batch_norm
        regularizer = layers.l2_regularizer(1.0)

        with self.graph.as_default():
            with tf.variable_scope('input'):
                # Build Inputs
                x, y = self.input_setup()  # Build Feature Extractor

            with tf.variable_scope('net'):
                with tf.variable_scope(self.extractor.__name__):
                    conv1_1_out, output = self.extractor(inputs=x,
                                                         reuse=False,
                                                         name="flow",
                                                         training=self.training,
                                                         activation_fn=activation_fn,
                                                         initializer=initializer,
                                                         keep_prob=keep_prob,
                                                         normalizer_fn=normalizer_fn,
                                                         normalizer_params=normalizer_params,
                                                         regularizer=regularizer)

                print("Visual Encoder - Output:", output.get_shape().as_list())
                with tf.variable_scope("dense"):
                    with tf.variable_scope("flatten"):
                        net = layers.flatten(output)
                        print("flattened - Output:", net.get_shape().as_list())

                    #with tf.variable_scope("fully1") as scope:
                    #    net = layers.fully_connected(net, 2048, activation_fn=tf.nn.leaky_relu, scope=scope)
                    #    print("fully1 - Output:", net.get_shape().as_list())

                    with tf.variable_scope("fully2") as scope:
                        y_logits = layers.fully_connected(net, self.classes, activation_fn=tf.nn.leaky_relu, scope=scope)
                        print("fully2 - Output:", y_logits.get_shape().as_list())

            # Define Loss
            with tf.name_scope('output'):
                loss = self.loss(y_logits, y)

            # Define Training Setup
            self.training_setup(loss)

            '''
            # Create the variable to store the images to output to TensorBoard
            with tf.name_scope('tensorboard_input'):
                tb_images = tf.placeholder(tf.float32, [None, self.input_shape[0]*self.input_shape[1]*self.input_shape[2]], name='tb_images')
            self.model["tb_images"] = tb_images
            '''

            # Merge all variables
            _, _ = self.merge()

            # Initialize Variables and save
            self.init_model()

    def evaluate(self, x_batch, y_batch, mode="train"):
        predictions, step = self.sess.run([self.model["predictions"], self.global_step],
                                          feed_dict={self.model["x"]: x_batch, self.model["y"]: y_batch})
        for i, logits in enumerate(predictions):
            string = self.directory + '/hist/' + mode + "_" + str(y_batch[i]) + '_prediction_' + str(i) + ".png"
            util.create_output_classification(img=x_batch[i], y=y_batch[i], y_logits=logits, classes=self.classnames,
                                              string=string)

    def load(self, path=None):
        print("Restore model from", path)
        if path is None:
            self.model["saver"].restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
            # self.get_model_from_graph(self.model_parameters)
        else:
            self.model["saver"].restore(self.sess, path)
            # self.get_model_from_graph(self.model_parameters)

    '''
    def get_misclassified_images(self, predictions_bool, predictions):
        """ Group all images that have been incorrectly classified into a tensor
            that can be read by TensorBoard.
        """
        wrong_predictions = [count for count, p in enumerate(predictions_bool) if not p]
        wrong_images = np.zeros((len(wrong_predictions), self.input_shape[0]*2, self.input_shape[1], self.input_shape[2]))

        for count, index in enumerate(wrong_predictions):
            correct_number = np.zeros(self.input_shape)
            cv2.putText(correct_number, str(predictions[index]), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # predicted_number = np.reshape(data.test.images[index], self.input_shape)
            predicted_number = np.zeros(self.input_shape)
            # TODO
            img = np.append(predicted_number, correct_number, axis=0)
            wrong_images[count] = 1-img

        print("%s images have been incorrectly classified." % len(wrong_predictions))
        image_summary_op = tf.summary.image('images', tf.reshape(wrong_images, [-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]]), 50)
        return image_summary_op, np.reshape(wrong_images, [-1, self.input_shape[0]*self.input_shape[1]*self.input_shape[2]])  # reshape to make sure it's right for the TB tensor
    '''



#Load the definitions of Inception-Resnet-v2 architecture
import tensorflow.contrib.slim as slim
from models.tfmodels.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope


class InceptionResnet(AbstractModel):
    def __init__(self, path, input_shape=(299, 299, 3), classnames=None, extractor=fe.inception_resnet_v2, training=False, learning_rate=1e-4, dropout=0.0, rewrite=False):
        self.name = "Classification_" + extractor.__name__ + '_d' + str(dropout) + '_l' + str(learning_rate)
        super(InceptionResnet, self).__init__(path, self.name, rewrite, learning_rate)
        self.input_shape = input_shape
        self.training = training
        self.extractor = extractor
        self.dropout = dropout
        self.classnames = classnames
        self.classes = len(classnames)
        self.model_parameters = ['x', 'y', 'loss', 'accuracy', 'global_step', 'predictions', 'predictions_op', 'correct_prediction_op', 'merged', 'optimizer']
        self.pretrained_weights = "/home/matthias/Schreibtisch/skincancer/data/pretrained_weights/inception_resnet_v2_2016_08_30.ckpt"

    def loss(self, output, labels):
            predictions = tf.nn.softmax(output)
            # Define the accuracy calculation
            with tf.name_scope('Accuracy'):
                predictions_op = tf.argmax(predictions, 1)
                correct_prediction_op = tf.equal(predictions_op, tf.to_int64(labels))
                accuracy_op = tf.reduce_mean(tf.cast(correct_prediction_op, tf.float32))
                tf.summary.scalar('Accuracy', accuracy_op)
                self.model["predictions"] = predictions
                self.model["predictions_op"] = predictions_op
                self.model["correct_prediction_op"] = correct_prediction_op
                self.model["accuracy"] = accuracy_op

            loss = me.sparse_xentropy_loss(output, labels)
            tf.summary.scalar('loss', loss)
            self.model["loss"] = loss
            return loss

    def inference(self):
        print("BUILDING GRAPH")
        keep_prob = 1.0 - self.dropout
        normalizer_params = {'is_training': self.training, 'updates_collections': None, 'decay': 0.99}
        initializer = layers.variance_scaling_initializer()
        activation_fn = tf.nn.relu
        normalizer_fn = layers.batch_norm
        regularizer = layers.l2_regularizer(1.0)

        with self.graph.as_default():
            # Create a placeholder to pass the input image
            img_tensor = tf.placeholder(tf.float32, shape=(None, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            self.model["x"] = img_tensor

            # Scale the image inputs to {+1, -1} from 0 to 255
            #img_scaled = tf.scalar_mul((1.0 / 255), img_tensor)
            img_scaled = tf.subtract(img_tensor, 0.5)
            img_scaled = tf.multiply(img_scaled, 2.0)

            # load Graph definitions
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(img_scaled, is_training=self.training)

            self.model["saver"] = tf.train.Saver()
            self.pretrained_init(self.pretrained_weights)

            # Initialize After Loading
            y = tf.placeholder(tf.int64, [None], name="y")
            self.model['y'] = y
            # Define Loss
            with tf.name_scope('output'):
                loss = self.loss(end_points["Logits"], y)

            # Define Training Setup
            self.training_setup(loss)

            # Merge all variables
            _, _ = self.merge()

            # Initialize Variables and save
            self.init_model()

    def evaluate(self, x_batch, y_batch, mode="train"):
        predictions, step = self.sess.run([self.model["predictions"], self.global_step],
                                          feed_dict={self.model["x"]: x_batch, self.model["y"]: y_batch})
        for i, logits in enumerate(predictions):
            string = self.directory + '/hist/' + mode + "_" + str(y_batch[i]) + '_prediction_' + str(i) + ".png"
            util.create_output_classification(img=x_batch[i], y=y_batch[i], y_logits=logits, classes=self.classnames,
                                              string=string)

    def pretrained_init(self, checkpoint_file):
        self.model["saver"].restore(self.sess, checkpoint_file)

    def predict(self, x_batch):
        pred_prob = self.sess.run(self.model["predictions"], feed_dict={self.model["x"]: x_batch})
        probabilities = pred_prob[0, 0:]
        print(probabilities)

    def load(self, path=None):
        self.pretrained_init(path)

        """
        print("Restore model from", path)
        if path is None:
            self.model["saver"].restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
            # self.get_model_from_graph(self.model_parameters)
        else:
            self.model["saver"].restore(self.sess, path)
            # self.get_model_from_graph(self.model_parameters)
        """
