import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

from collections import namedtuple

'''
MODELS:
custom
CNN
inception_v3_base
CNN_Resnet
CNN_7x7
CNN3x3
CNN_Nvidia
CNN3x3_orig
CNN3x3_orig_3x3_pool
CNN3x3_orig_spatial_dropout
'''


def custom(inputs, reuse=False, name="flow", training=True,
           activation_fn=tf.nn.relu,
           initializer=layers.variance_scaling_initializer(),
           keep_prob=1.0,
           normalizer_fn=layers.batch_norm,
           normalizer_params={},
           regularizer=layers.l2_regularizer(1.0)):
    """ Custom CNN """
    print("Custom Feature Extractor:", name, " Input:", inputs.get_shape().as_list())
    with tf.variable_scope("conv1") as scope:
        conv1_1_out = tf.contrib.layers.conv2d(inputs, 64, [3, 3], stride=[1, 1], activation_fn=tf.nn.relu, padding='Valid',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        print("conv1 - Output:", conv1_1_out.get_shape().as_list())

    with tf.variable_scope("pool1"):
        net = tf.contrib.layers.max_pool2d(conv1_1_out, [2, 2], stride=[2, 2], padding='SAME')
        print("pool1 - Output:", net.get_shape().as_list())

    with tf.variable_scope("conv2") as scope:
        net = tf.contrib.layers.conv2d(net, 64, [3, 3], stride=[1, 1], activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        print("conv2 - Output:", net.get_shape().as_list())

    with tf.variable_scope("pool2"):
        net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=[2, 2], padding='SAME')
        print("pool2 - Output:", net.get_shape().as_list())

    with tf.variable_scope("conv3") as scope:
        net = tf.contrib.layers.conv2d(net, 32, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        print("conv3 - Output:", net.get_shape().as_list())

    with tf.variable_scope("conv4") as scope:
        net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        print("conv4 - Output:", net.get_shape().as_list())

    with tf.variable_scope("pool3"):
        net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=[2, 2], padding='SAME')
        print("pool3 - Output:", net.get_shape().as_list())

    with tf.variable_scope("conv5") as scope:
        conv3_2_out = tf.contrib.layers.conv2d(net, 16, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        print("conv5 - Output:", net.get_shape().as_list())

    return conv1_1_out, conv3_2_out


def CNN(inputs, reuse=False, name="flow", training=True,
        activation_fn=tf.nn.relu,
        initializer=layers.variance_scaling_initializer(),
        keep_prob=1.0,
        normalizer_fn=layers.batch_norm,
        normalizer_params={},
        regularizer=layers.l2_regularizer(1.0),
        final_endpoint='Mixed_7c',
        min_depth=16,
        depth_multiplier=1.0,
        scope=None):
    padding = 'VALID'
    with tf.variable_scope('conv1_1') as scope:
        conv1_1_out = layers.conv2d(inputs, 24, [5, 5],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv1_1_out = layers.max_pool2d(conv1_1_out, [2, 2], padding=padding)
    with tf.variable_scope('conv2_1') as scope:
        conv2_1_out = layers.conv2d(conv1_1_out, 36, [5, 5],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv2_1_out = layers.max_pool2d(conv2_1_out, [2, 2], padding=padding)
    with tf.variable_scope('conv2_2') as scope:
        conv2_2_out = layers.conv2d(conv2_1_out, 48, [5, 5],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv3_1') as scope:
        conv3_1_out = layers.conv2d(conv2_2_out, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv3_2') as scope:
        conv3_2_out = layers.conv2d(conv3_1_out, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope)
        print('FC input:' + repr(conv3_2_out))
    return conv1_1_out, conv3_2_out


def inception_v3_base(inputs, reuse=False, name="flow", training=True,
                      activation_fn=tf.nn.relu,
                      initializer=layers.variance_scaling_initializer(),
                      keep_prob=1.0,
                      normalizer_fn=layers.batch_norm,
                      normalizer_params={},
                      regularizer=layers.l2_regularizer(1.0),
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
    """Inception model from http://arxiv.org/abs/1512.00567.
    Constructs an Inception v3 network from inputs to the given final endpoint.
    This method can construct the network up to the final inception block
    Mixed_7c.
    Note that the names of the layers in the paper do not correspond to the names
    of the endpoints registered by this function although they build the same
    network.
    Here is a mapping from the old_names to the new names:
    Old name          | New name
    =======================================
    conv0             | Conv2d_1a_3x3
    conv1             | Conv2d_2a_3x3
    conv2             | Conv2d_2b_3x3
    pool1             | MaxPool_3a_3x3
    conv3             | Conv2d_3b_1x1
    conv4             | Conv2d_4a_3x3
    pool2             | MaxPool_5a_3x3
    mixed_35x35x256a  | Mixed_5b
    mixed_35x35x288a  | Mixed_5c
    mixed_35x35x288b  | Mixed_5d
    mixed_17x17x768a  | Mixed_6a
    mixed_17x17x768b  | Mixed_6b
    mixed_17x17x768c  | Mixed_6c
    mixed_17x17x768d  | Mixed_6d
    mixed_17x17x768e  | Mixed_6e
    mixed_8x8x1280a   | Mixed_7a
    mixed_8x8x2048a   | Mixed_7b
    mixed_8x8x2048b   | Mixed_7c
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
        'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
        'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
        'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      scope: Optional variable_scope.
    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.
    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.

    images = tf.image.resize_images(inputs, size=(299, 299))
    end_points = {}
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with tf.variable_scope(scope, 'InceptionV3', [images]):
        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='VALID'):
                # 299 x 299 x 3
                end_point = 'Conv2d_1a_3x3'
                net = slim.conv2d(images, depth(32), [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
                # 149 x 149 x 32
                end_point = 'Conv2d_2a_3x3'
                net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
                # 147 x 147 x 32
                end_point = 'Conv2d_2b_3x3'
                net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
                # 147 x 147 x 64
                end_point = 'MaxPool_3a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
                # 73 x 73 x 64
                end_point = 'Conv2d_3b_1x1'
                net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
                # 73 x 73 x 80.
                end_point = 'Conv2d_4a_3x3'
                net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
                # 71 x 71 x 192.
                end_point = 'MaxPool_5a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
                # 35 x 35 x 192.

            # Inception blocks
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # mixed: 35 x 35 x 256.
                end_point = 'Mixed_5b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                               scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net

                # mixed_1: 35 x 35 x 288.
                end_point = 'Mixed_5c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                               scope='Conv_1_0c_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                               scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net

                # mixed_2: 35 x 35 x 288.
                end_point = 'Mixed_5d'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                               scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net

                # mixed_3: 17 x 17 x 768.
                end_point = 'Mixed_6a'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                               padding='VALID', scope='Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                               scope='Conv2d_0b_3x3')
                        branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                               padding='VALID', scope='Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net

                # mixed4: 17 x 17 x 768.
                end_point = 'Mixed_6b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                               scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                               scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                               scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net

                # mixed_5: 17 x 17 x 768.
                end_point = 'Mixed_6c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                               scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                               scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                               scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
                # mixed_6: 17 x 17 x 768.
                end_point = 'Mixed_6d'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                               scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                               scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                               scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net

                # mixed_7: 17 x 17 x 768.
                end_point = 'Mixed_6e'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                               scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                               scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net

                # mixed_8: 8 x 8 x 1280.
                end_point = 'Mixed_7a'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                               padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                        branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                               padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
                # mixed_9: 8 x 8 x 2048.
                end_point = 'Mixed_7b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat(axis=3, values=[
                            slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                            slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(
                            branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = tf.concat(axis=3, values=[
                            slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                            slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(
                            branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net

                # mixed_10: 8 x 8 x 2048.
                end_point = 'Mixed_7c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat(axis=3, values=[
                            slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                            slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(
                            branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = tf.concat(axis=3, values=[
                            slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                            slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(
                            branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                    net = slim.avg_pool2d(net, [8, 8], stride=8, scope='AvgPool_8x8')
                end_points[end_point] = net
                if end_point == final_endpoint: return end_points, net
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CNN_Resnet(inputs, reuse=False, name="flow", training=True,
               activation_fn=tf.nn.relu,
               initializer=layers.variance_scaling_initializer(),
               keep_prob=1.0,
               normalizer_fn=layers.batch_norm,
               normalizer_params={},
               regularizer=layers.l2_regularizer(1.0)):
    with tf.variable_scope('conv1_1') as scope:
        conv1_1_out = layers.conv2d(inputs, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='SAME',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv1_1_pooled = layers.max_pool2d(conv1_1_out, [2, 2], stride=2, padding='SAME')
    with tf.variable_scope('conv1_connection') as scope:
        connection_conv1 = layers.conv2d(conv1_1_pooled, 64, [1, 1],
                                         activation_fn=activation_fn,
                                         padding='SAME',
                                         weights_regularizer=regularizer,
                                         weights_initializer=initializer,
                                         normalizer_fn=normalizer_fn,
                                         normalizer_params=normalizer_params, scope=scope, stride=1)
        connection_conv1 = layers.max_pool2d(connection_conv1, [2, 2], stride=2, padding='SAME')
    with tf.variable_scope('conv2_1') as scope:
        conv2_1_out = layers.conv2d(conv1_1_pooled, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='SAME',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope)
    with tf.variable_scope('conv2_2') as scope:
        conv2_2_out = layers.conv2d(conv2_1_out, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='SAME',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope)
        conv2_2_out = layers.max_pool2d(conv2_2_out, [2, 2], stride=2, padding='SAME')
        conv2_2_added = conv2_2_out + connection_conv1
    with tf.variable_scope('conv2_connection') as scope:
        connection_conv2 = layers.conv2d(conv2_2_added, 128, [1, 1],
                                         activation_fn=activation_fn,
                                         padding='SAME',
                                         weights_regularizer=regularizer,
                                         weights_initializer=initializer,
                                         normalizer_fn=normalizer_fn,
                                         normalizer_params=normalizer_params, scope=scope, stride=1)
        connection_conv2 = layers.max_pool2d(connection_conv2, [2, 2], stride=2, padding='SAME')
    with tf.variable_scope('conv3_1') as scope:
        conv3_1_out = layers.conv2d(conv2_2_added, 128, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='SAME',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv3_2') as scope:
        conv3_2_out = layers.conv2d(conv3_1_out, 128, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='SAME',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope)
        conv3_2_out = layers.max_pool2d(conv3_2_out, [2, 2], stride=2, padding='SAME')
        conv3_2_added = conv3_2_out + connection_conv2
    with tf.variable_scope('conv3_connection') as scope:
        connection_conv3 = layers.conv2d(conv3_2_added, 256, [1, 1],
                                         activation_fn=activation_fn,
                                         padding='SAME',
                                         weights_regularizer=regularizer,
                                         weights_initializer=initializer,
                                         normalizer_fn=normalizer_fn,
                                         normalizer_params=normalizer_params, scope=scope, stride=1)
        connection_conv3 = layers.max_pool2d(connection_conv3, [2, 2], stride=2, padding='SAME')
    with tf.variable_scope('conv4_1') as scope:
        conv4_1_out = layers.conv2d(conv3_2_added, 256, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='SAME',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv4_2') as scope:
        conv4_2_out = layers.conv2d(conv4_1_out, 256, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='SAME',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope)
        conv4_2_out = layers.max_pool2d(conv4_2_out, [2, 2], stride=2, padding='SAME')
        conv4_2_added = conv4_2_out + connection_conv3
    with tf.variable_scope('conv4_connection') as scope:
        connection_conv4 = layers.conv2d(conv4_2_added, 512, [1, 1],
                                         activation_fn=activation_fn,
                                         padding='SAME',
                                         weights_regularizer=regularizer,
                                         weights_initializer=initializer,
                                         normalizer_fn=normalizer_fn,
                                         normalizer_params=normalizer_params, scope=scope, stride=1)
        connection_conv4 = layers.max_pool2d(connection_conv4, [2, 2], stride=2, padding='SAME')
    with tf.variable_scope('conv5_1') as scope:
        conv5_1_out = layers.conv2d(conv4_2_added, 512, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='SAME',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv5_2') as scope:
        conv5_2_out = layers.conv2d(conv5_1_out, 512, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='SAME',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope)
        conv5_2_out = layers.max_pool2d(conv5_2_out, [2, 2], stride=2, padding='SAME')
        conv5_out = conv5_2_out + connection_conv4
    return conv1_1_out, conv5_out


def CNN_7x7(inputs, reuse=False, name="flow", training=True,
            activation_fn=tf.nn.relu,
            initializer=layers.variance_scaling_initializer(),
            keep_prob=1.0,
            normalizer_fn=layers.batch_norm,
            normalizer_params={},
            regularizer=layers.l2_regularizer(1.0)):
    with tf.variable_scope('conv1_1') as scope:
        conv1_1_out = layers.conv2d(inputs, 64, [7, 7],
                                    activation_fn=activation_fn,
                                    padding='VALID',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv1_1_out = layers.max_pool2d(conv1_1_out, [2, 2])
    with tf.variable_scope('conv2_1') as scope:
        conv2_1_out = layers.conv2d(conv1_1_out, 64, [5, 5],
                                    activation_fn=activation_fn,
                                    padding='VALID',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv2_1_out = layers.max_pool2d(conv2_1_out, [2, 2])
    with tf.variable_scope('conv2_2') as scope:
        conv2_2_out = layers.conv2d(conv2_1_out, 64, [5, 5],
                                    activation_fn=activation_fn,
                                    padding='VALID',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv3_1') as scope:
        conv3_1_out = layers.conv2d(conv2_2_out, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='VALID',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv3_2') as scope:
        conv3_2_out = layers.conv2d(conv3_1_out, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding='VALID',
                                    weights_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope)
        print('FC input:' + repr(conv3_2_out))
    return conv1_1_out, conv3_2_out


def CNN3x3(inputs, reuse=False, name="flow", training=True,
           activation_fn=tf.nn.relu,
           initializer=layers.variance_scaling_initializer(),
           keep_prob=1.0,
           normalizer_fn=layers.batch_norm,
           normalizer_params={},
           regularizer=layers.l2_regularizer(1.0)):
    padding = 'SAME'
    with tf.variable_scope('conv1_1') as scope:
        conv1_1_out = layers.conv2d(inputs, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv1_1_out = layers.max_pool2d(conv1_1_out, [2, 2], padding=padding)
    with tf.variable_scope('conv1_2') as scope:
        conv1_2_out = layers.conv2d(conv1_1_out, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv1_2_out = layers.max_pool2d(conv1_2_out, [2, 2], padding=padding)
    with tf.variable_scope('conv1_3') as scope:
        conv1_3_out = layers.conv2d(conv1_2_out, 64, [3, 3],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv1_3_out = layers.max_pool2d(conv1_3_out, [2, 2], padding=padding)
    with tf.variable_scope('conv2_1') as scope:
        conv2_1_out = layers.conv2d(conv1_3_out, 128, [3, 3],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv2_1_out = layers.max_pool2d(conv2_1_out, [2, 2], padding=padding)
    with tf.variable_scope('conv2_2') as scope:
        conv2_2_out = layers.conv2d(conv2_1_out, 128, [3, 3],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        conv2_2_out = layers.max_pool2d(conv2_2_out, [2, 2], padding=padding)
    with tf.variable_scope('conv3_1') as scope:
        conv3_1_out = layers.conv2d(conv2_2_out, 256, [3, 3],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv3_2') as scope:
        conv3_2_out = layers.conv2d(conv3_1_out, 256, [3, 3],
                                    activation_fn=activation_fn,
                                    padding=padding,
                                    weights_regularizer=regularizer,
                                    biases_regularizer=regularizer,
                                    weights_initializer=initializer,
                                    normalizer_fn=normalizer_fn,
                                    normalizer_params=normalizer_params, scope=scope, stride=1)
        print('FC input:' + repr(conv3_2_out))
    return conv1_2_out, conv3_2_out


def CNN_Nvidia(inputs, reuse=False, name="flow", training=True,
               activation_fn=tf.nn.relu,
               initializer=layers.variance_scaling_initializer(),
               keep_prob=1.0,
               normalizer_fn=layers.batch_norm,
               normalizer_params={},
               regularizer=layers.l2_regularizer(1.0)):
    padding = 'VALID'
    images = tf.transpose(inputs, [0, 3, 1, 2])
    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([layers.conv2d], data_format='NCHW'):
        with tf.variable_scope('conv1_1') as scope:
            conv1_1_out = layers.conv2d(images, 24, [5, 5],
                                        activation_fn=activation_fn,
                                        padding=padding,
                                        weights_regularizer=regularizer,
                                        biases_regularizer=regularizer,
                                        weights_initializer=initializer,
                                        normalizer_fn=normalizer_fn,
                                        normalizer_params=normalizer_params, scope=scope, stride=2)
        with tf.variable_scope('conv2_1') as scope:
            conv2_1_out = layers.conv2d(conv1_1_out, 36, [5, 5],
                                        activation_fn=activation_fn,
                                        padding=padding,
                                        weights_regularizer=regularizer,
                                        biases_regularizer=regularizer,
                                        weights_initializer=initializer,
                                        normalizer_fn=normalizer_fn,
                                        normalizer_params=normalizer_params, scope=scope, stride=2)
        with tf.variable_scope('conv2_2') as scope:
            conv2_2_out = layers.conv2d(conv2_1_out, 48, [5, 5],
                                        activation_fn=activation_fn,
                                        padding=padding,
                                        weights_regularizer=regularizer,
                                        biases_regularizer=regularizer,
                                        weights_initializer=initializer,
                                        normalizer_fn=normalizer_fn,
                                        normalizer_params=normalizer_params, scope=scope, stride=2)
        with tf.variable_scope('conv3_1') as scope:
            conv3_1_out = layers.conv2d(conv2_2_out, 64, [3, 3],
                                        activation_fn=activation_fn,
                                        padding='SAME',
                                        weights_regularizer=regularizer,
                                        biases_regularizer=regularizer,
                                        weights_initializer=initializer,
                                        normalizer_fn=normalizer_fn,
                                        normalizer_params=normalizer_params, scope=scope, stride=1)
        with tf.variable_scope('conv3_2') as scope:
            conv3_2_out = layers.conv2d(conv3_1_out, 64, [3, 3],
                                        activation_fn=activation_fn,
                                        padding=padding,
                                        weights_regularizer=regularizer,
                                        biases_regularizer=regularizer,
                                        weights_initializer=initializer,
                                        normalizer_fn=normalizer_fn,
                                        normalizer_params=normalizer_params, scope=scope)
            print('FC input:' + repr(conv3_2_out))
    return tf.transpose(conv1_1_out, [0, 2, 3, 1]), conv3_2_out


def CNN3x3_orig(inputs, reuse=False, name="flow", training=True,
                activation_fn=tf.nn.relu,
                initializer=layers.variance_scaling_initializer(),
                keep_prob=1.0,
                normalizer_fn=layers.batch_norm,
                normalizer_params={},
                regularizer=layers.l2_regularizer(1.0)):
    padding = 'VALID'
    images = tf.transpose(inputs, [0, 3, 1, 2])
    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([layers.conv2d, layers.max_pool2d], data_format='NCHW'):
        with tf.variable_scope('conv1_1') as scope:
            conv1_out = layers.conv2d(images, 24, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
        with tf.variable_scope('conv1_2') as scope:
            conv1_out = layers.conv2d(conv1_out, 24, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv1_out = layers.max_pool2d(conv1_out, [2, 2], padding=padding)
        with tf.variable_scope('conv2_1') as scope:
            conv2_out = layers.conv2d(conv1_out, 36, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
        with tf.variable_scope('conv2_2') as scope:
            conv2_out = layers.conv2d(conv2_out, 36, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv2_out = layers.max_pool2d(conv2_out, [2, 2], padding=padding)
        with tf.variable_scope('conv3_1') as scope:
            conv3_out = layers.conv2d(conv2_out, 48, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
        with tf.variable_scope('conv3_2') as scope:
            conv3_out = layers.conv2d(conv3_out, 48, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv3_out = layers.max_pool2d(conv3_out, [2, 2], padding='SAME')
        with tf.variable_scope('conv4') as scope:
            conv4_out = layers.conv2d(conv3_out, 64, [3, 3],
                                      activation_fn=activation_fn,
                                      padding='SAME',
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
        with tf.variable_scope('conv5') as scope:
            conv5_out = layers.conv2d(conv4_out, 64, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope)
            print('FC input:' + repr(conv5_out))
    return tf.transpose(conv1_out, [0, 2, 3, 1]), conv5_out


def CNN3x3_orig_3x3_pool(inputs, reuse=False, name="flow", training=True,
                         activation_fn=tf.nn.relu,
                         initializer=layers.variance_scaling_initializer(),
                         keep_prob=1.0,
                         normalizer_fn=layers.batch_norm,
                         normalizer_params={},
                         regularizer=layers.l2_regularizer(1.0)):
    padding = 'VALID'
    with tf.variable_scope('conv1_1') as scope:
        conv1_out = layers.conv2d(inputs, 24, [3, 3],
                                  activation_fn=activation_fn,
                                  padding=padding,
                                  weights_regularizer=regularizer,
                                  biases_regularizer=regularizer,
                                  weights_initializer=initializer,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv1_2') as scope:
        conv1_out = layers.conv2d(conv1_out, 24, [3, 3],
                                  activation_fn=activation_fn,
                                  padding=padding,
                                  weights_regularizer=regularizer,
                                  biases_regularizer=regularizer,
                                  weights_initializer=initializer,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params, scope=scope, stride=1)
        conv1_out = layers.max_pool2d(conv1_out, [3, 3], padding=padding)
    with tf.variable_scope('conv2_1') as scope:
        conv2_out = layers.conv2d(conv1_out, 36, [3, 3],
                                  activation_fn=activation_fn,
                                  padding=padding,
                                  weights_regularizer=regularizer,
                                  biases_regularizer=regularizer,
                                  weights_initializer=initializer,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv2_2') as scope:
        conv2_out = layers.conv2d(conv2_out, 36, [3, 3],
                                  activation_fn=activation_fn,
                                  padding=padding,
                                  weights_regularizer=regularizer,
                                  biases_regularizer=regularizer,
                                  weights_initializer=initializer,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params, scope=scope, stride=1)
        conv2_out = layers.max_pool2d(conv2_out, [3, 3], padding=padding)
    with tf.variable_scope('conv3_1') as scope:
        conv3_out = layers.conv2d(conv2_out, 48, [3, 3],
                                  activation_fn=activation_fn,
                                  padding=padding,
                                  weights_regularizer=regularizer,
                                  biases_regularizer=regularizer,
                                  weights_initializer=initializer,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv3_2') as scope:
        conv3_out = layers.conv2d(conv3_out, 48, [3, 3],
                                  activation_fn=activation_fn,
                                  padding=padding,
                                  weights_regularizer=regularizer,
                                  biases_regularizer=regularizer,
                                  weights_initializer=initializer,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params, scope=scope, stride=1)
        conv3_out = layers.max_pool2d(conv3_out, [3, 3], padding='SAME')
    with tf.variable_scope('conv4') as scope:
        conv4_out = layers.conv2d(conv3_out, 64, [3, 3],
                                  activation_fn=activation_fn,
                                  padding='SAME',
                                  weights_regularizer=regularizer,
                                  biases_regularizer=regularizer,
                                  weights_initializer=initializer,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params, scope=scope, stride=1)
    with tf.variable_scope('conv5') as scope:
        conv5_out = layers.conv2d(conv4_out, 64, [3, 3],
                                  activation_fn=activation_fn,
                                  padding=padding,
                                  weights_regularizer=regularizer,
                                  biases_regularizer=regularizer,
                                  weights_initializer=initializer,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params, scope=scope)
        print('FC input:' + repr(conv5_out))
    return conv1_out, conv5_out


def CNN3x3_orig_spatial_dropout(inputs, reuse=False, name="flow", training=True,
                                activation_fn=tf.nn.relu,
                                initializer=layers.variance_scaling_initializer(),
                                keep_prob=1.0,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params={},
                                regularizer=layers.l2_regularizer(1.0),
                                final_endpoint='Mixed_7c',
                                min_depth=16,
                                depth_multiplier=1.0,
                                scope=None):
    padding = 'VALID'
    images = tf.transpose(inputs, [0, 3, 1, 2])
    arg_scope = tf.contrib.framework.arg_scope
    images = tf.nn.dropout(images, keep_prob=keep_prob)
    with arg_scope([layers.conv2d, layers.max_pool2d], data_format='NCHW'):
        with tf.variable_scope('conv1_1') as scope:
            conv1_out = layers.conv2d(images, 24, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv1_out = spatial_dropout(conv1_out, keep_prob)

        with tf.variable_scope('conv1_2') as scope:
            conv1_out = layers.conv2d(conv1_out, 24, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv1_out = layers.max_pool2d(conv1_out, [2, 2], padding=padding)

        with tf.variable_scope('conv2_1') as scope:
            conv2_out = layers.conv2d(conv1_out, 36, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv2_out = spatial_dropout(conv2_out, keep_prob)

        with tf.variable_scope('conv2_2') as scope:
            conv2_out = layers.conv2d(conv2_out, 36, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv2_out = layers.max_pool2d(conv2_out, [2, 2], padding=padding)

        with tf.variable_scope('conv3_1') as scope:
            conv3_out = layers.conv2d(conv2_out, 48, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv3_out = spatial_dropout(conv2_out, keep_prob)

        with tf.variable_scope('conv3_2') as scope:
            conv3_out = layers.conv2d(conv3_out, 48, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv3_out = layers.max_pool2d(conv3_out, [2, 2], padding='SAME')

        with tf.variable_scope('conv4') as scope:
            conv4_out = layers.conv2d(conv3_out, 64, [3, 3],
                                      activation_fn=activation_fn,
                                      padding='SAME',
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope, stride=1)
            conv4_out = spatial_dropout(conv4_out, keep_prob)

        with tf.variable_scope('conv5') as scope:
            conv5_out = layers.conv2d(conv4_out, 64, [3, 3],
                                      activation_fn=activation_fn,
                                      padding=padding,
                                      weights_regularizer=regularizer,
                                      biases_regularizer=regularizer,
                                      weights_initializer=initializer,
                                      normalizer_fn=normalizer_fn,
                                      normalizer_params=normalizer_params, scope=scope)
            conv5_out = spatial_dropout(conv5_out, keep_prob)
            print('FC input:' + repr(conv5_out))
    return tf.transpose(conv1_out, [0, 2, 3, 1]), conv5_out


def spatial_dropout(x, keep_prob):
    """
    Spatial dropout layer that drops out entire feature maps instead of individual pixels.
    Code is from https://stats.stackexchange.com/questions/282282/how-is-spatial-dropout-in-2d-implemented

    x is a convnet activation with shape BxWxHxF where F is the
    number of feature maps for that layer
    keep_prob is the proportion of feature maps we want to keep :param x:

    :param x:
    :param keep_prob:
    :return:
    """

    # get the batch size and number of feature maps
    num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]

    # get some uniform noise between keep_prob and 1 + keep_prob
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(num_feature_maps,
                                       dtype=x.dtype)

    # if we take the floor of this, we get a binary matrix where
    # (1-keep_prob)% of the values are 0 and the rest are 1
    binary_tensor = tf.floor(random_tensor)

    # Reshape to multiply our feature maps by this tensor correctly
    binary_tensor = tf.reshape(binary_tensor,
                               [-1, 1, 1, tf.shape(x)[3]])
    # Zero out feature maps where appropriate; scale up to compensate
    ret = tf.div(x, keep_prob) * binary_tensor
    return ret
