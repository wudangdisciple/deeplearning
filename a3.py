import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os
os.environ['TF_CPP_LOG_LEVEL']='2'#关闭警告信息
#x:[60k,28,28]
#y:[60k]
(x,y),_ = datasets.mnist.load_data()
#转换成tensor
x = tf.convert_to_tensor(x,dtype=tf.float32)/255#将0~255中的值归一化到0~1
y = tf.convert_to_tensor(y,dtype=tf.int32)

print(x.shape,y.shape,x.dtype,y.dtype)
print(tf.reduce_min(x),tf.reduce_max(x))#查看x的最小值和最大值
print(tf.reduce_min(y),tf.reduce_max(y))#查看y的最小值和最大值
train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)#随机抽取128批
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:',sample[0].shape,sample[1].shape)
#创建权值[b,784]->[b,256]->[b,128]->[b,10]
#[dim_in,dim_out],[dim_out]
w1 = tf.random.truncated_normal([784,256])
b1 = tf.zeros([256])