import sys
import tensorflow as tf
import keras as ks
print("tensorflow v", tf.__version__)
print("keras v", ks.__version__)
print("python version", sys.version)
noGpu = len(tf.config.list_physical_devices('GPU'))
print("gpu is ", "available" if noGpu else "not available")

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print('GPU is available and GPU number(s) :', len(gpus))

# Auto run on CPU or GPU
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Run on the GPU
c = tf.matmul(a, b)
print(c)