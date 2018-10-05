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
y_data=np.exp(np.sin(np.multiply(6,x_data)))

#define placeholders for the inputs
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#add hidden layer
l1=add_layer(xs,1,100,activation_function=tf.nn.relu)
l2=add_layer(l1,100,100,activation_function=tf.nn.relu)
l3=add_layer(l2,100,100,activation_function=tf.nn.relu)

prediction=add_layer(l3,100,1,activation_function=None)
#Error
loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-y_data),reduction_indices=[1]))

params = tf.trainable_variables()
gradients = tf.gradients(loss, params)
gradient_norm = tf.global_norm(gradients)

train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)


fig=plt.figure()
ax=fig.add_subplot(2,1,1)
bx=fig.add_subplot(2,1,2)

for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%4==0:
        gn = sess.run(gradient_norm, feed_dict={xs:x_data, ys:y_data})
        loss_ = sess.run(loss, feed_dict={xs:x_data, ys:y_data})
        print(gn,loss_)
        ax.scatter(i,gn,c='b')
        bx.scatter(i,loss_,c='b')

plt.show()
print("variable count",np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
input()
