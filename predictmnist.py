import tensorflow as tf

import inputdata

import numpy as np

import os

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

def getGray(image_file):
    tmpls=[]
    for h in range(0,  image_file.size[1]):#h
        for w in range(0, image_file.size[0]):#w
            tmpls.append( image_file.getpixel((w,h)))

    return tmpls

def getAvg(ls):#获取平均灰度值
    return sum(ls)/len(ls)

def getPicArray(filename):
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    im = im.convert("L")
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    #im_arr = np.array(out.convert('L'))
    im_arr = np.array(out)

    grayls = getGray(out)#灰度集合
    avg = getAvg(grayls)#灰度平均值

    # print(im_arr)
    num0 = 0
    num255 = 0
    threshold = 100
    # print(avg)

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1

    if num255 > num0:
        # print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if im_arr[x][y] < threshold:
                    im_arr[x][y] = 0
                else:
                    im_arr[x][y] = 255

    out = Image.fromarray(np.uint8(im_arr))
    savePath = os.getcwd()+"/convert/"
    if not os.path.exists(savePath) :
       os.mkdir(savePath)
    out.save(savePath + os.path.basename(filename))
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

ckpt_dir = "ckpt"
image_dir = "images"

for root, dirs, files in os.walk(image_dir):
    print(root)
    print(dirs)
    print(files)

with tf.Session() as sess:
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)

    for i in range(len(files)):
        filename = image_dir + "/" + files[i]
        x_img = getPicArray(filename)
        # print(x_img)
        output = sess.run(y_conv, feed_dict={x: x_img, keep_prob: 1.0})
        # print("the y_con : %s " % (output))
        print("%s, the predict is : %d " % (files[i], np.argmax(output)))
