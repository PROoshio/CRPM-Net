import tensorflow as tf 
from tensorflow.python.ops import array_ops

def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 5.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 5.0))
    return tf.reduce_sum(per_entry_cross_ent,-1)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def max_pool(x,n,k=2,padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, k, k, 1], padding=padding)


def complex_cross_dilated_conv(input_real,input_imag,scope_name,input_shape,keep_prob,regularizer=None,rate=2):
    print(scope_name)
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE):
        conv_weight_real = tf.get_variable(
                name="weight_real",
                shape=input_shape,
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
        conv_weight_imag = tf.get_variable(
                name="weight_imag",
                shape=input_shape,
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
        conv_bias_real = tf.get_variable(
                name="bias_real",
                shape=[input_shape[-1]],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)
        conv_bias_imag = tf.get_variable(
                name="bias_imag",
                shape=[input_shape[-1]],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)
        real_part = tf.nn.atrous_conv2d(input_real,conv_weight_real,rate,padding="SAME")
        cross_real_part = tf.nn.atrous_conv2d(input_real,conv_weight_imag,rate,padding="SAME")
        imag_part = tf.nn.atrous_conv2d(input_imag,conv_weight_imag,rate,padding="SAME")
        cross_imag_part = tf.nn.atrous_conv2d(input_imag,conv_weight_real,rate,padding="SAME")
        conv_real = tf.subtract(real_part,imag_part)
        conv_imag = tf.add(cross_real_part,cross_imag_part)
        relu_real = tf.nn.relu(tf.nn.bias_add(conv_real,conv_bias_real))
        relu_imag = tf.nn.relu(tf.nn.bias_add(conv_imag,conv_bias_imag))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(conv_weight_real))
            tf.add_to_collection('losses',regularizer(conv_weight_imag))
            tf.add_to_collection('losses',regularizer(conv_bias_real))
            tf.add_to_collection('losses',regularizer(conv_bias_imag))
        return tf.nn.dropout(relu_real,keep_prob),tf.nn.dropout(relu_imag,keep_prob)


def complex_cross_conv(input_real,input_imag,scope_name,input_shape,keep_prob,padding,regularizer=None):
    print(scope_name)
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE):
        conv_weight_real = tf.get_variable(
                name="weight_real",
                shape=input_shape,
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
        conv_weight_imag = tf.get_variable(
                name="weight_imag",
                shape=input_shape,
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
        conv_bias_real = tf.get_variable(
                name="bias_real",
                shape=[input_shape[-1]],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)
        conv_bias_imag = tf.get_variable(
                name="bias_imag",
                shape=[input_shape[-1]],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)
        real_part = tf.nn.conv2d(input_real,conv_weight_real,strides=[1,1,1,1],padding=padding)
        cross_real_part = tf.nn.conv2d(input_real,conv_weight_imag,strides=[1,1,1,1],padding=padding)
        imag_part = tf.nn.conv2d(input_imag,conv_weight_imag,strides=[1,1,1,1],padding=padding)
        cross_imag_part = tf.nn.conv2d(input_imag,conv_weight_real,strides=[1,1,1,1],padding=padding)
        conv_real = tf.subtract(real_part,imag_part)
        conv_imag = tf.add(cross_real_part,cross_imag_part)
        relu_real = tf.nn.relu(tf.nn.bias_add(conv_real,conv_bias_real))
        relu_imag = tf.nn.relu(tf.nn.bias_add(conv_imag,conv_bias_imag))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(conv_weight_real))
            tf.add_to_collection('losses',regularizer(conv_weight_imag))
            tf.add_to_collection('losses',regularizer(conv_bias_real))
            tf.add_to_collection('losses',regularizer(conv_bias_imag))
        return tf.nn.dropout(relu_real,keep_prob),tf.nn.dropout(relu_imag,keep_prob)

def complex_cross_deconv(input_real,input_imag,scope_name,input_shape,regularizer=None):
    with tf.variable_scope(scope_name+"deconv2d",reuse=tf.AUTO_REUSE):
        deconv_weight_real = tf.get_variable(
                name="weight_real",
                shape=input_shape,
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
        deconv_weight_imag = tf.get_variable(
                name="weight_imag",
                shape=input_shape,
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
        deconv_bias_real = tf.get_variable(
                name="bias_real",
                shape=[input_shape[-2]],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)
        deconv_bias_imag = tf.get_variable(
                name="bias_imag",
                shape=[input_shape[-2]],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)
        x_shape = tf.shape(input_real)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, input_shape[-2]])
        trans_real = tf.nn.conv2d_transpose(input_real, deconv_weight_real, output_shape, strides=[1, 2, 2, 1], padding='SAME', name="trans_real")
        trans_imag = tf.nn.conv2d_transpose(input_imag, deconv_weight_imag, output_shape, strides=[1, 2, 2, 1], padding='SAME', name="trans_imag")
        trans_cross_real = tf.nn.conv2d_transpose(input_real, deconv_weight_imag, output_shape, strides=[1, 2, 2, 1], padding='SAME', name="trans_cross_real")
        trans_cross_imag = tf.nn.conv2d_transpose(input_imag, deconv_weight_real, output_shape, strides=[1, 2, 2, 1], padding='SAME', name="trans_cross_imag")
        real = tf.subtract(trans_real,trans_imag)
        imag = tf.add(trans_cross_real,trans_cross_imag)
        relu_real = tf.nn.relu(real+deconv_bias_real)
        relu_imag = tf.nn.relu(imag+deconv_bias_imag)
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(deconv_weight_real))
            tf.add_to_collection('losses',regularizer(deconv_weight_imag))
            tf.add_to_collection('losses',regularizer(deconv_bias_real))
            tf.add_to_collection('losses',regularizer(deconv_bias_imag))
        return relu_real,relu_imag

def complex_cross_fc(input_real,input_imag,scope_name,input_shape,isActive):
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE):
        fc_weight_real = tf.get_variable(
                name="fc_weight_real",
                shape=input_shape,
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
        fc_weight_imag = tf.get_variable(
                name="fc_weight_imag",
                shape=input_shape,
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
        fc_bias_real = tf.get_variable(
                name="fc_bias_real",
                shape=[input_shape[-1]],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)
        fc_bias_imag = tf.get_variable(
                name="fc_bias_imag",
                shape=[input_shape[-1]],
                initializer=tf.zeros_initializer(),
                dtype=tf.float32)
        real_part = tf.matmul(input_real,fc_weight_real)
        cross_real_part = tf.matmul(input_real,fc_weight_imag)
        imag_part = tf.matmul(input_imag,fc_weight_imag)
        cross_imag_part = tf.matmul(input_imag,fc_weight_real)
        fc_real = tf.subtract(real_part,imag_part)
        fc_imag = tf.add(real_part,imag_part)
        if isActive:
            relu_real = tf.nn.relu(tf.nn.bias_add(fc_real,fc_bias_real),'real')
            relu_imag = tf.nn.relu(tf.nn.bias_add(fc_imag,fc_bias_imag),'imag')
        else:
            relu_real = tf.nn.bias_add(fc_real,fc_bias_real)
            relu_imag = tf.nn.bias_add(fc_imag,fc_bias_imag)
        return relu_real,relu_imag


def prelu(inputs,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable("alphas_prelu", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs-abs(inputs))*0.5
        return pos + neg