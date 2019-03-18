import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import cv2
import os
import numpy as np
from scipy import misc

tf.app.flags.DEFINE_string('server', 'localhost:8500','PredictionService host:port')
tf.app.flags.DEFINE_string('image', '/Users/yeppy/TestProject/TestData/4233.jpg', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

host, port = FLAGS.server.split(':')
channel = grpc.insecure_channel('127.0.0.1', int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'facenet'
request.model_spec.signature_name = 'calculate_embeddings'


im = misc.imread("/Users/yeppy/TestProject/TestData/4233.jpg")
print(im.shape[0])
print(im.shape[1])
print(im.shape[2])

im_resized=cv2.resize(im,(160,160),interpolation=cv2.INTER_CUBIC)

request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(im_resized, shape=[1, im_resized.shape[0], im_resized.shape[1], im_resized.shape[2]], dtype=tf.float32))
request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
result = stub.Predict(request, 1000.0) 
response = np.array(result.outputs['embeddings'].float_val)
print(response)
print(response.shape)
print(type(response))


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'facenet'
  request.model_spec.signature_name = 'calculate_embeddings'
  im = misc.imread(os.path.expanduser("/home/vivek/a.jpg"), mode='RGB')
  print(im.shape[0])
  print(im.shape[1])
  print(im.shape[2])
  request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(im, shape=[1, im.shape[0], im.shape[1], im.shape[2]], dtype=tf.float32))
  request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  response = np.array(result.outputs['embeddings'].float_val)
  print(response)
  print(response.shape)
  print(type(response))
if __name__ == '__main__':
  tf.app.run()