#!/usr/bin/env python2.7
# import cifar10
from __future__ import print_function

import os
import random
import signal
import sys
import time
from threading import Thread

import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow.python.saved_model import utils
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '',
                           'path to image in JPEG format')



images = [os.path.join(FLAGS.image, f) for f in os.listdir(FLAGS.image) if
            os.path.isfile(os.path.join(FLAGS.image, f))]
image_data = []
for img in images:
    f = open(img, 'rb')
    image_data.append(f.read())
host, port = FLAGS.server.split(':')
random.seed(1)
host, port = FLAGS.server.split(':')
channel = implementations.insecure_channel(host, int(port))
tstub = prediction_service_pb2.beta_create_PredictionService_stub(
    channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'inception'
request.model_spec.signature_name = 'predict_images'
request.inputs['images'].CopyFrom(
    tf.contrib.util.make_tensor_proto(random.choice(image_data), shape=[1]))


def run_inference():
    while True:
        try:
            #print("Request sent.")
            res=tstub.Predict(request, 99999999999)
            #print(res.outputs)
            return
        except Exception as e:
            print(e)#, end='\r'
            time.sleep(0)
            pass

def main(_):
    average_interarrival_time = 0.07
    number_of_requests = 100
    random_seed=1
    random.seed(random_seed)
    interval_arrival_time = [random.expovariate(1.0 / interarrival) for _ in range(number_of_request)]

    while True:
        print("Start " + str(number_of_requests) + " request.")
        episode_start_time = time.time()
        threads_list=[]
        for i in range(number_of_requests):
            time.sleep(interval_arrival_time[i])
            thrd = Thread(target=run_inference)
            thrd.start()
            threads_list.append(thrd)
        print("All request sent. waiting for completion.")
        for t in threads_list:
            t.join()
            # print(t)
        print("All request finished.")
        print(time.time() - episode_start_time)
        time.sleep(10)#wait for server restart
      # timeout
    


if __name__ == '__main__':
    tf.app.run()
