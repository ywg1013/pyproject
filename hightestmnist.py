import tensorflow as tf

import inputdata


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


# MNIST数据输入
mnist = inputdata.read_data_sets("MNIST_data/", one_hot=True)


#mnist = learn.datasets.load_dataset('mnist')

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

y_ = tf.placeholder("float", [None, 10])

# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
# 启动创建的模型，并初始化变量
saver = tf.train.Saver()
total_step = 30000;

isTrain = True
ckpt_dir = "ckpt"

with tf.Session() as sess:
    sess.run(init)
    if isTrain:
        for i in range(total_step):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                # print("step %d, test accuracy %g" % (i, accuracy.eval(session=sess,feed_dict={
                #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        saver_path = saver.save(sess, ckpt_dir + "/model.ckpt", global_step=total_step)  # 将模型保存到save/model.ckpt文件
        print("Model saved in file:", saver_path)
    else:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

    print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


