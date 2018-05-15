import numpy as np
import tensorflow as tf

batch_size = 300
num_epochs = 200

train_data = np.load('/home/minhnd/Dropbox/train_data.npy').reshape((55000, 28, 28, 1))
train_labels = np.load('/home/minhnd/Dropbox/train_labels.npy')
eval_data = np.load('/home/minhnd/Dropbox/eval_data.npy').reshape((10000, 28, 28, 1))
eval_labels = np.load('/home/minhnd/Dropbox/eval_labels.npy')

print('MNIST training data:', type(train_data), train_data.shape)
print('MNIST training labels:', type(train_labels), train_labels.shape)
print('MNIST evaluation data:', type(eval_data), eval_data.shape)
print('MNIST evaluation labels:', type(eval_labels), eval_labels.shape)

weight_init = tf.contrib.layers.xavier_initializer()
bias_init = tf.zeros_initializer()

def build_graph(input_layer):
    conv1_W = tf.get_variable(name='conv1_W', initializer=weight_init, shape=(5, 5, 1, 32))
    conv1_b = tf.get_variable(name='conv1_b', initializer=bias_init, shape=(32))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_layer, conv1_W, strides=[1, 1, 1, 1], padding="SAME"), conv1_b))
    
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2_W = tf.get_variable(name='conv2_W', initializer=weight_init, shape=(5, 5, 32, 64))
    conv2_b = tf.get_variable(name='conv2_b', initializer=bias_init, shape=(64))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, conv2_W, strides=[1, 1, 1, 1], padding="SAME"), conv2_b))
    
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    fc_W = tf.get_variable(name='fc_W', initializer=weight_init, shape=(7, 7, 64, 1024))
    fc_b = tf.get_variable(name='fc_b', initializer=bias_init, shape=(1024))
    fc = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool2, fc_W, strides=[1, 1, 1, 1], padding="VALID"), fc_b))
    
    dropout = tf.nn.dropout(fc, keep_prob=0.6)
    logits_W = tf.get_variable(name='logits_W', initializer=weight_init, shape=(1, 1, 1024, 10))
    logits_b = tf.get_variable(name='logits_b', initializer=bias_init, shape=(10))
    logits = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout, logits_W, strides=[1, 1, 1, 1], padding="VALID"), logits_b))

    return tf.squeeze(logits, [1, 2])

def _input():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input')
    labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
    return x, labels

def _loss(logits, labels):
    cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cross_entropy_loss

def _train_op(loss, global_step):
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss, global_step=global_step)
    return train_step

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    global_step = tf.train.get_or_create_global_step()
    x, labels = _input()
    logits = build_graph(x)
    loss = _loss(logits, labels)
    train_step = _train_op(loss, global_step)
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    writer = tf.summary.FileWriter('./summary/')
    writer.add_graph(sess.graph)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merge_summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    train_idx = np.arange(train_data.shape[0])
    eval_idx = np.arange(eval_data.shape[0])
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        print('Epoch %d' % epoch)

        np.random.shuffle(train_idx)
        train_data = train_data[train_idx]
        train_labels = train_labels[train_idx]

        np.random.shuffle(eval_idx)
        eval_data = eval_data[eval_idx]
        eval_labels = eval_labels[eval_idx]

        mean_loss = []
        mean_acc = []
        num_batch = int(len(train_idx) // batch_size)
        for batch in range(num_batch):
            print('Training on batch %d / %d' % (batch, num_batch), end='\r')
            top = batch * batch_size
            bot = min((batch + 1) * batch_size, len(train_idx))
            img_batch = train_data[top:bot]
            label_batch = train_labels[top:bot]
            ttl, _, acc, s = sess.run([loss, train_step, accuracy, merge_summary], feed_dict={x: img_batch, labels: label_batch})
            writer.add_summary(s, int(global_step.eval()))
            mean_loss.append(ttl)
            mean_acc.append(acc)
        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        print('\nTraining loss: %f' % mean_loss)
        print('Training accuracy: %f' % mean_acc)

        mean_loss = []
        mean_acc = []
        num_batch = int(len(eval_idx) // batch_size)
        for batch in range(num_batch):
            top = batch * batch_size
            bot = min((batch + 1) * batch_size, len(eval_idx))
            img_batch = eval_data[top:bot]
            label_batch = eval_labels[top:bot]
            ttl, acc = sess.run([loss, accuracy], feed_dict={x: img_batch, labels: label_batch})
            mean_loss.append(ttl)
            mean_acc.append(acc)
        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        print('\nTesting loss: %f' % mean_loss)
        print('Testing accuracy: %f' % mean_acc)

    saver = tf.train.Saver()
    saver.save(sess, './mnist_model/mnist_recogniser')
