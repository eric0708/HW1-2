from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import pylab as pl
import matplotlib.pyplot as plt

xs=tf.placeholder(tf.float32,[None, 784])
ys=tf.placeholder(tf.float32,[None, 10])
p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)
#weightrecord=tf.placeholder(tf.float32,[None,784*1000+1000*600+600*10])

 
weights1=tf.Variable(tf.random_normal([784,1000],stddev=0.01))
weights2=tf.Variable(tf.random_normal([1000,600],stddev=0.01))
weights3=tf.Variable(tf.random_normal([600,10],stddev=0.01))
 
 
def model(xs,weights1,weights2,weights3,p_keep_input,p_keep_hidden):
    xs= tf.nn.dropout(xs,p_keep_input)
 
    h1 = tf.nn.relu(tf.matmul(xs,weights1))
    h1 = tf.nn.dropout(h1,p_keep_hidden)

    h2 = tf.nn.relu(tf.matmul(h1,weights2))
    h2 = tf.nn.dropout(h2,p_keep_hidden)
 
    return tf.matmul(h2,weights3)
 
 
py_x = model(xs,weights1,weights2,weights3,p_keep_input,p_keep_hidden)


lossfunction=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=ys))
train= tf.train.RMSPropOptimizer(0.001, 0.9).minimize(lossfunction)


accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(py_x, 1),tf.argmax(ys,1)),tf.float32))


epoch=200
batchsize=10
keep_input=0.8
keep_hidden=0.75
 


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    step=0
    for i in range(epoch):
        step+=1
        batch_x,batch_y=mnist.train.next_batch(batchsize)
        sess.run(train,feed_dict={xs: batch_x,ys: batch_y,p_keep_input:keep_input,p_keep_hidden:keep_hidden})
        if step%3==0:
        	weights1_1D=np.reshape(sess.run(weights1),(1,784*1000))
        	weights2_1D=np.reshape(sess.run(weights2),(1,1000*600))
        	weights3_1D=np.reshape(sess.run(weights3),(1,600*10))
        	weightholder=weights2_1D
        	#weightholder=np.hstack((weights1_1D,weights2_1D,weights3_1D))
        	if step/3==1:
        		weightrecord=weightholder
        	else:
        		weightrecord=np.vstack((weightrecord,weightholder))
        	loss,acc = sess.run([lossfunction,accuracy],feed_dict={xs:batch_x,ys:batch_y, p_keep_input: 1.,p_keep_hidden:1.})
        	print("Epoch: {}".format(step), "\tLoss: {:.6f}".format(loss), "\tTraining Accuracy: {:.5f}".format(acc))
    weightrecord=weightrecord.T
    pca = PCA(n_components=2)
    pca.fit(weightrecord)
    weightrecord=pca.transform(weightrecord)
    x=pca.components_[0][:]
    y=pca.components_[1][:]
    print("Training Accuracy: {:0.5f}".format(sess.run(accuracy, feed_dict={xs: mnist.train.images, ys: mnist.train.labels, p_keep_input: 1., p_keep_hidden: 1.})))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    step=0
    for i in range(epoch):
        step+=1
        batch_x,batch_y=mnist.train.next_batch(batchsize)
        sess.run(train,feed_dict={xs: batch_x,ys: batch_y,p_keep_input:keep_input,p_keep_hidden:keep_hidden})
        if step%3==0:
        	weights1_1D=np.reshape(sess.run(weights1),(1,784*1000))
        	weights2_1D=np.reshape(sess.run(weights2),(1,1000*600))
        	weights3_1D=np.reshape(sess.run(weights3),(1,600*10))
        	weightholder=weights2_1D
        	#weightholder=np.hstack((weights1_1D,weights2_1D,weights3_1D))
        	if step/3==1:
        		weightrecord=weightholder
        	else:
        		weightrecord=np.vstack((weightrecord,weightholder))
        	loss,acc = sess.run([lossfunction,accuracy],feed_dict={xs:batch_x,ys:batch_y, p_keep_input: 1.,p_keep_hidden:1.})
        	print("Epoch: {}".format(step), "\tLoss: {:.6f}".format(loss), "\tTraining Accuracy: {:.5f}".format(acc))
    weightrecord=weightrecord.T
    pca = PCA(n_components=2)
    pca.fit(weightrecord)
    weightrecord=pca.transform(weightrecord)
    x1=pca.components_[0][:]
    y1=pca.components_[1][:]
    print("Training Accuracy: {:0.5f}".format(sess.run(accuracy, feed_dict={xs: mnist.train.images, ys: mnist.train.labels, p_keep_input: 1., p_keep_hidden: 1.})))
    

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    step=0
    for i in range(epoch):
        step+=1
        batch_x,batch_y=mnist.train.next_batch(batchsize)
        sess.run(train,feed_dict={xs: batch_x,ys: batch_y,p_keep_input:keep_input,p_keep_hidden:keep_hidden})
        if step%3==0:
        	weights1_1D=np.reshape(sess.run(weights1),(1,784*1000))
        	weights2_1D=np.reshape(sess.run(weights2),(1,1000*600))
        	weights3_1D=np.reshape(sess.run(weights3),(1,600*10))
        	weightholder=weights2_1D
        	#weightholder=np.hstack((weights1_1D,weights2_1D,weights3_1D))
        	if step/3==1:
        		weightrecord=weightholder
        	else:
        		weightrecord=np.vstack((weightrecord,weightholder))
        	loss,acc = sess.run([lossfunction,accuracy],feed_dict={xs:batch_x,ys:batch_y, p_keep_input: 1.,p_keep_hidden:1.})
        	print("Epoch: {}".format(step), "\tLoss: {:.6f}".format(loss), "\tTraining Accuracy: {:.5f}".format(acc))
    weightrecord=weightrecord.T
    pca = PCA(n_components=2)
    pca.fit(weightrecord)
    weightrecord=pca.transform(weightrecord)
    x2=pca.components_[0][:]
    y2=pca.components_[1][:]
    print("Training Accuracy: {:0.5f}".format(sess.run(accuracy, feed_dict={xs: mnist.train.images, ys: mnist.train.labels, p_keep_input: 1., p_keep_hidden: 1.})))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    step=0
    for i in range(epoch):
        step+=1
        batch_x,batch_y=mnist.train.next_batch(batchsize)
        sess.run(train,feed_dict={xs: batch_x,ys: batch_y,p_keep_input:keep_input,p_keep_hidden:keep_hidden})
        if step%3==0:
        	weights1_1D=np.reshape(sess.run(weights1),(1,784*1000))
        	weights2_1D=np.reshape(sess.run(weights2),(1,1000*600))
        	weights3_1D=np.reshape(sess.run(weights3),(1,600*10))
        	weightholder=weights2_1D
        	#weightholder=np.hstack((weights1_1D,weights2_1D,weights3_1D))
        	if step/3==1:
        		weightrecord=weightholder
        	else:
        		weightrecord=np.vstack((weightrecord,weightholder))
        	loss,acc = sess.run([lossfunction,accuracy],feed_dict={xs:batch_x,ys:batch_y, p_keep_input: 1.,p_keep_hidden:1.})
        	print("Epoch: {}".format(step), "\tLoss: {:.6f}".format(loss), "\tTraining Accuracy: {:.5f}".format(acc))
    weightrecord=weightrecord.T
    pca = PCA(n_components=2)
    pca.fit(weightrecord)
    weightrecord=pca.transform(weightrecord)
    x3=pca.components_[0][:]
    y3=pca.components_[1][:]
    print("Training Accuracy: {:0.5f}".format(sess.run(accuracy, feed_dict={xs: mnist.train.images, ys: mnist.train.labels, p_keep_input: 1., p_keep_hidden: 1.})))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    step=0
    for i in range(epoch):
        step+=1
        batch_x,batch_y=mnist.train.next_batch(batchsize)
        sess.run(train,feed_dict={xs: batch_x,ys: batch_y,p_keep_input:keep_input,p_keep_hidden:keep_hidden})
        if step%3==0:
        	weights1_1D=np.reshape(sess.run(weights1),(1,784*1000))
        	weights2_1D=np.reshape(sess.run(weights2),(1,1000*600))
        	weights3_1D=np.reshape(sess.run(weights3),(1,600*10))
        	weightholder=weights2_1D
        	#weightholder=np.hstack((weights1_1D,weights2_1D,weights3_1D))
        	if step/3==1:
        		weightrecord=weightholder
        	else:
        		weightrecord=np.vstack((weightrecord,weightholder))
        	loss,acc = sess.run([lossfunction,accuracy],feed_dict={xs:batch_x,ys:batch_y, p_keep_input: 1.,p_keep_hidden:1.})
        	print("Epoch: {}".format(step), "\tLoss: {:.6f}".format(loss), "\tTraining Accuracy: {:.5f}".format(acc))
    weightrecord=weightrecord.T
    pca = PCA(n_components=2)
    pca.fit(weightrecord)
    weightrecord=pca.transform(weightrecord)
    x4=pca.components_[0][:]
    y4=pca.components_[1][:]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    step=0
    for i in range(epoch):
        step+=1
        batch_x,batch_y=mnist.train.next_batch(batchsize)
        sess.run(train,feed_dict={xs: batch_x,ys: batch_y,p_keep_input:keep_input,p_keep_hidden:keep_hidden})
        if step%3==0:
        	weights1_1D=np.reshape(sess.run(weights1),(1,784*1000))
        	weights2_1D=np.reshape(sess.run(weights2),(1,1000*600))
        	weights3_1D=np.reshape(sess.run(weights3),(1,600*10))
        	weightholder=weights2_1D
        	#weightholder=np.hstack((weights1_1D,weights2_1D,weights3_1D))
        	if step/3==1:
        		weightrecord=weightholder
        	else:
        		weightrecord=np.vstack((weightrecord,weightholder))
        	loss,acc = sess.run([lossfunction,accuracy],feed_dict={xs:batch_x,ys:batch_y, p_keep_input: 1.,p_keep_hidden:1.})
        	print("Epoch: {}".format(step), "\tLoss: {:.6f}".format(loss), "\tTraining Accuracy: {:.5f}".format(acc))
    weightrecord=weightrecord.T
    pca = PCA(n_components=2)
    pca.fit(weightrecord)
    weightrecord=pca.transform(weightrecord)
    x5=pca.components_[0][:]
    y5=pca.components_[1][:]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    step=0
    for i in range(epoch):
        step+=1
        batch_x,batch_y=mnist.train.next_batch(batchsize)
        sess.run(train,feed_dict={xs: batch_x,ys: batch_y,p_keep_input:keep_input,p_keep_hidden:keep_hidden})
        if step%3==0:
        	weights1_1D=np.reshape(sess.run(weights1),(1,784*1000))
        	weights2_1D=np.reshape(sess.run(weights2),(1,1000*600))
        	weights3_1D=np.reshape(sess.run(weights3),(1,600*10))
        	weightholder=weights2_1D
        	#weightholder=np.hstack((weights1_1D,weights2_1D,weights3_1D))
        	if step/3==1:
        		weightrecord=weightholder
        	else:
        		weightrecord=np.vstack((weightrecord,weightholder))
        	loss,acc = sess.run([lossfunction,accuracy],feed_dict={xs:batch_x,ys:batch_y, p_keep_input: 1.,p_keep_hidden:1.})
        	print("Epoch: {}".format(step), "\tLoss: {:.6f}".format(loss), "\tTraining Accuracy: {:.5f}".format(acc))
    weightrecord=weightrecord.T
    pca = PCA(n_components=2)
    pca.fit(weightrecord)
    weightrecord=pca.transform(weightrecord)
    x6=pca.components_[0][:]
    y6=pca.components_[1][:]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    step=0
    for i in range(epoch):
        step+=1
        batch_x,batch_y=mnist.train.next_batch(batchsize)
        sess.run(train,feed_dict={xs: batch_x,ys: batch_y,p_keep_input:keep_input,p_keep_hidden:keep_hidden})
        if step%3==0:
        	weights1_1D=np.reshape(sess.run(weights1),(1,784*1000))
        	weights2_1D=np.reshape(sess.run(weights2),(1,1000*600))
        	weights3_1D=np.reshape(sess.run(weights3),(1,600*10))
        	weightholder=weights2_1D
        	#weightholder=np.hstack((weights1_1D,weights2_1D,weights3_1D))
        	if step/3==1:
        		weightrecord=weightholder
        	else:
        		weightrecord=np.vstack((weightrecord,weightholder))
        	loss,acc = sess.run([lossfunction,accuracy],feed_dict={xs:batch_x,ys:batch_y, p_keep_input: 1.,p_keep_hidden:1.})
        	print("Epoch: {}".format(step), "\tLoss: {:.6f}".format(loss), "\tTraining Accuracy: {:.5f}".format(acc))
    weightrecord=weightrecord.T
    pca = PCA(n_components=2)
    pca.fit(weightrecord)
    weightrecord=pca.transform(weightrecord)
    x7=pca.components_[0][:]
    y7=pca.components_[1][:]


plt.scatter(x,y,c='r')
plt.scatter(x1,y1,c='g')
plt.scatter(x2,y2,c='b')
plt.scatter(x3,y3,c='c')
plt.scatter(x4,y4,c='m')
plt.scatter(x5,y5,c='y')
plt.scatter(x6,y6,c='k')
plt.scatter(x7,y7,c='brown')
plt.show()
    



