import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size], stddev=0.1))
    biases=tf.Variable(tf.zeros([1,out_size]))
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
  
#create data
x_data=np.linspace(-1,1,300)[:,np.newaxis]
y_data=np.divide(np.sin(np.multiply(10,x_data)),np.multiply(10,x_data))

#define placeholders for the inputs
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#add hidden layer
Weights1=tf.Variable(tf.random_normal([1,100], stddev=0.1))
biases1=tf.Variable(tf.zeros([1,100]))
l1=tf.nn.relu(tf.matmul(xs,Weights1)+biases1)

Weights2=tf.Variable(tf.random_normal([100,100], stddev=0.1))
biases2=tf.Variable(tf.zeros([1,100]))
l2=tf.nn.relu(tf.matmul(l1,Weights2)+biases2)

Weights3=tf.Variable(tf.random_normal([100,1], stddev=0.1))
biases3=tf.Variable(tf.zeros([1,1]))
prediction=(tf.matmul(l2,Weights3)+biases3)

#Error
loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-y_data),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(0.1).minimize(loss)

params = tf.trainable_variables()
gradients = tf.gradients(loss, params)
gradient_norm = tf.global_norm(gradients)

#run the model with gven weights and biases
Weights12=tf.placeholder(tf.float32,[1,100])
biases12=tf.placeholder(tf.float32,[1,100])
l12=tf.nn.relu(tf.matmul(xs,Weights12)+biases12)

Weights22=tf.placeholder(tf.float32,[100,100])
biases22=tf.placeholder(tf.float32,[1,100])
l22=tf.nn.relu(tf.matmul(l1,Weights2)+biases2)

Weights32=tf.placeholder(tf.float32,[100,1])
biases32=tf.placeholder(tf.float32,[1,1])
prediction2=(tf.matmul(l22,Weights32)+biases32)
loss2=tf.reduce_mean(tf.reduce_sum(tf.square(prediction2-y_data),reduction_indices=[1]))

train_step2=tf.train.AdamOptimizer(0.00001).minimize(gradient_norm)
print("loss,minimal ratial")
for n in range(100):

	init=tf.initialize_all_variables()
	sess=tf.Session()
	sess.run(init)


	fig=plt.figure()
	ax=fig.add_subplot(2,1,1)
	bx=fig.add_subplot(2,1,2)

	for i in range(4000):
		if i<2000:
			sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
		else:
			sess.run(train_step2,feed_dict={xs:x_data,ys:y_data})

		if i%4==0:
			gn = sess.run(gradient_norm, feed_dict={xs:x_data, ys:y_data})
			loss_ = sess.run(loss, feed_dict={xs:x_data, ys:y_data})
	Weights1save = sess.run(Weights1, feed_dict={xs:x_data, ys:y_data})
	biases1save = sess.run(biases1, feed_dict={xs:x_data, ys:y_data})
	Weights2save = sess.run(Weights2, feed_dict={xs:x_data, ys:y_data})
	biases2save = sess.run(biases2, feed_dict={xs:x_data, ys:y_data})
	Weights3save = sess.run(Weights3, feed_dict={xs:x_data, ys:y_data})
	biases3save = sess.run(biases3, feed_dict={xs:x_data, ys:y_data})
	initial_loss = loss_

	k=0.0
	for j in range(1000):
		Weights1random = Weights1save + np.random.normal(loc=0.0, scale=0.0001, size=(1,100))
		Weights2random = Weights2save + np.random.normal(loc=0.0, scale=0.0001, size=(100,100))
		Weights3random = Weights3save + np.random.normal(loc=0.0, scale=0.0001, size=(100,1))
		loss2_ = sess.run(loss2, feed_dict={xs:x_data, ys:y_data, Weights12:Weights1random, biases12:biases1save, biases22:biases2save, biases32:biases3save, Weights22:Weights2random, Weights32:Weights3random})

		if loss2_>initial_loss:
			k+=1
	print(initial_loss,k/1000)

print("variable count",np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

