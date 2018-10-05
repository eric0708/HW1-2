import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE



#create data
LR=0.01
x=np.linspace(-1,1,300)[:,np.newaxis]

y=np.divide(np.sin(np.multiply(10,x)),np.multiply(10,x))



#define placeholders for the inputs
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#add hidden layer
Weights1=tf.Variable(tf.random_normal([1,3], stddev=0.1))
biases1=tf.Variable(tf.zeros([1,3]))
l1=tf.nn.relu(tf.matmul(xs,Weights1)+biases1)

Weights2=tf.Variable(tf.random_normal([3,3], stddev=0.1))
biases2=tf.Variable(tf.zeros([1,3]))
l2=tf.nn.relu(tf.matmul(l1,Weights2)+biases2)

Weights3=tf.Variable(tf.random_normal([3,1], stddev=0.1))
biases3=tf.Variable(tf.zeros([1,1]))
prediction=(tf.matmul(l2,Weights3)+biases3)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)







loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(LR).minimize(loss)




for i in range(1000):
    sess.run(train_step,feed_dict={xs:x,ys:y})
    Weights1_1D=np.reshape(sess.run(Weights1),(1,3))
    Weights2_1D=np.reshape(sess.run(Weights2),(1,9))
    Weights3_1D=np.reshape(sess.run(Weights3),(1,3))
    a=sess.run(loss,feed_dict={xs:x,ys:y})
    X=np.hstack((Weights1_1D,Weights2_1D,Weights3_1D))
    if i==0 :
        Y=X
        H=a
    else :
        Y=np.vstack((Y,X))
        H=np.vstack((H,a))
        
tsne = TSNE(n_components=2)
tsne.fit_transform(Y)
print(tsne.embedding_)
print (H)

c,d=np.split(tsne.embedding_, 2, axis=1)
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(c,d,H)
plt.show()


    

	

