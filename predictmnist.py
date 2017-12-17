import tensorflow as tf

import inputdata

import numpy as np

from PIL import Image

import cv2 as cv

from tensorflow.contrib import learn


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def getPicArray(filename):
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    im_arr = np.array(out.convert('L'))

    num0 = 0
    num255 = 0
    threshold = 100

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1

    if (num255 > num0):
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if (im_arr[x][y] < threshold):  im_arr[x][y] = 0

    #    out = Image.fromarray(np.uint8(im_arr))
    #    out.save(filename.split('/')[0] + '/28pix/' + filename.split('/')[1])
    # print im_arr
    nm = im_arr.reshape((1, 784))
    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)

    return nm


# MNIST数据输入
mnist = inputdata.read_data_sets("MNIST_data/", one_hot=True)

# mnist = learn.datasets.load_dataset('mnist')

x = tf.placeholder(tf.float32, [None, 784])  # 图像输入向量
W = tf.Variable(tf.zeros([784, 10]))  # 权重，初始化值为全零
b = tf.Variable(tf.zeros([10]))  # 偏置，初始化值为全零

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 进行模型计算，y_conv是预测，y_ 是实际
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# y = tf.nn.softmax(tf.matmul(x, W) + b)

init = tf.global_variables_initializer()
# 启动创建的模型，并初始化变量
saver = tf.train.Saver()

isTrain = False
ckpt_dir = "ckpt"

with tf.Session() as sess:
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)

    filename = "images/1.jpg"
    img = np.array(Image.open(filename))
    im = cv.imread(filename, cv.IMREAD_GRAYSCALE).astype(np.float32)
    im = cv.resize(im, (28, 28), interpolation=cv.INTER_CUBIC)
    # 图片预处理
    # img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype(np.float32)
    # 数据从0~255转为-0.5~0.5
    img_gray = (im - (255 / 2.0)) / 255
    # cv.imshow('out',img_gray)
    # cv2.waitKey(0)
    x_img = np.reshape(img_gray, [-1, 784])

    x_img = getPicArray(filename)
    # print(x_img)
    output = sess.run(y_conv, feed_dict={x: x_img, keep_prob: 1.0})
    print("the y_con : %s " % (output))
    print("the predict is : %d " % (np.argmax(output)))
