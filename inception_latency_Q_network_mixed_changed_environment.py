from __future__ import print_function
import time
import subprocess
import os
import sys
import signal
import os.path
import psutil
import numpy as np
import random
import tensorflow as tf
import socket
import shutil

from interarrival_generator import *
from set_system_cpu_cores import *
data_floder = "raw_test_data/inception_Q_network_changeenvironment/"
recorded_training_feature_filename = data_floder + "recorded_training_feature.npy"
recorded_training_output_filename = data_floder + "recorded_training_output.npy"

preheat_time = 30
parameter_adjust_interval = 120
beta = 0.5
alpha = 0.3
value_table_size = [11,11,11]
explorational_rate = 0.1
display_step = 100
learning_rate = 0.7
training_interval = 0
max_training_failure = 10
max_training_step = 30000
try:
    recorded_training_feature = np.load(recorded_training_feature_filename)
    recorded_training_feature = recorded_training_feature.tolist()
except:
    recorded_training_feature = []
try:
    recorded_training_output = np.load(recorded_training_output_filename)
    recorded_training_output = recorded_training_output.tolist()
except:
    recorded_training_output = []

# Create model
def neural_net(x):
    layer_1 = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
    layer_2 = tf.contrib.layers.fully_connected(layer_1, 64, activation_fn=tf.nn.relu)
    #layer_3 = tf.contrib.layers.fully_connected(layer_2, n_hidden_3,
    #activation_fn=tf.nn.relu)

    out_layer = tf.contrib.layers.fully_connected(layer_2, 1, activation_fn=None)
    return out_layer

X = tf.placeholder("float", [None, 4], name="feature")
Y = tf.placeholder("float", [None, 1], name="label")
prediction = neural_net(X)
loss_op = tf.losses.huber_loss(labels=Y, predictions=prediction,delta=100.0)
train_op = tf.contrib.layers.optimize_loss(loss_op, tf.train.get_global_step(), optimizer='Ftrl',learning_rate=learning_rate)
init = tf.global_variables_initializer()

def update_value_matrix_with_policy_network(recorded_training_feature, recorded_training_output,interarrival_time):
    recorded_training_output = np.expand_dims(recorded_training_output, axis=1) / 1000
    recorded_training_feature = np.asarray(recorded_training_feature)
    #sess = tf.InteractiveSession()
    with tf.Session() as sess:
        # tf Graph input
        sess.run(init)
        best_loss = float("inf")
        training_failure_count = 0
        step = 0
        while training_failure_count < max_training_failure and step < max_training_step:
            step = step + 1
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: recorded_training_feature, Y: recorded_training_output})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss = sess.run(loss_op,feed_dict={X: recorded_training_feature, Y: recorded_training_output})
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))
                if best_loss > loss + training_interval:
                    best_loss = loss
                    training_failure_count = 0
                else:
                    training_failure_count = training_failure_count + 1
        print("Network training Finished!")
        value_matrix = np.random.random(value_table_size)
        ti = len(value_matrix)
        tj = len(value_matrix[0])
        tk = len(value_matrix[0][0])
        for i in range(ti):
            for j in range(tj):
                for k in range(tk):
                    tmp_input = np.asarray([interarrival_time, i, j, k])
                    tmp_input = np.expand_dims(tmp_input, axis=0)
                    value_matrix[i, j, k] = sess.run(prediction, feed_dict={X: tmp_input}) * 1000
    return value_matrix

def delete_server_output_files():
    try:
        os.remove(output_folder + '/direct_session.txt')
        os.remove(output_folder + '/main.txt')
        os.remove(output_folder + '/main_start.txt')
        os.remove(output_folder + '/main_end.txt')
        os.remove(output_folder + '/direct_session_start.txt')
        os.remove(output_folder + '/direct_session_end.txt')
    except:
        pass

delete_server_output_files()

def running_server_and_collect_result(intel, intra, number_of_thread):
    FNULL = open(os.devnull, 'w')
    if number_of_thread == 0:
        model_server_process = subprocess.call([binary_folder + '/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server',
                '--port=9000', '--enable_batching=false', '--model_name=inception',
                '--model_base_path=',
                '--tensorflow_session_parallelism_intro=' + str(intel),
                '--tensorflow_session_parallelism_intra=' + str(intra)],stdout=FNULL,stderr=FNULL)#
    else:
        model_server_process = subprocess.call([binary_folder + '/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server',
                '--port=9000', '--enable_batching=true', '--model_name=inception',
                '--model_base_path=',
                '--tensorflow_session_parallelism_intro=' + str(intel),
                '--tensorflow_session_parallelism_intra=' + str(intra),
                '--allowed_size=1', '--num_batch_threads=' + str(number_of_thread), '--batch_timeout=0'],stdout=FNULL,stderr=FNULL)#
    #print("Model server shut down.")
    return 0

def get_mean_latency_from_output_file():
    latency_data = np.loadtxt(output_folder + '/main.txt')
    assert latency_data.size==number_of_request
    t = np.mean(latency_data[-1 * (number_of_request + 1):-1])
    return t



def move_and_rename_server_output_files(i,j,k,count):
    
    if not os.path.exists(data_floder):
        os.makedirs(data_floder)
    
    os.rename(output_folder + '/main.txt',data_floder + "main_" + str(count) + ".txt")
    os.rename(output_folder + '/direct_session.txt',data_floder + "direct_session_" + str(count) + ".txt")
    os.rename(output_folder + '/main_start.txt',data_floder + "main_start_" + str(count) + ".txt")
    os.rename(output_folder + '/main_end.txt',data_floder + "main_end_" + str(count) + ".txt")
    os.rename(output_folder + '/direct_session_start.txt',data_floder + "direct_session_start_" + str(count) + ".txt")
    os.rename(output_folder + '/direct_session_end.txt',data_floder + "direct_session_end_" + str(count) + ".txt")

#number_of_request = 100
#parameter_alpha = 0.6


output_folder = "."
binary_folder = "."

def create_system_interaccival_file(interval_arrival_time):
    f = open(os.path.join(output_folder,"interarrival_time_generated.txt"),"w")
    for line in interval_arrival_time:
        f.writelines(float_to_string(line) + '\n')
    f.close()
    return

def main():
    set_system_cpu_cores(10)
    if not os.path.exists(data_floder):
        os.makedirs(data_floder)
    if psutil.cpu_count() != 10:
        print("CPU number error!!!!!!!")
        exit(-1)
    print(data_floder)
    shutil.copy(__file__, data_floder) 
    delete_server_output_files()
    try:
        Q_value_matrix = np.load(data_floder + "Q-matrix.npy")
        count = np.load(data_floder + "global_step.npy")
        accumulated_performance = np.load(data_floder + "accumulated_performance.npy")
        flag_initial_matrix = np.load(data_floder + "flag_initial_matrix.npy")
        recorded_training_feature = np.load(data_floder + "recorded_training_feature.npy")
        recorded_training_feature = recorded_training_feature.tolist()
        recorded_training_output = np.load(data_floder + "recorded_training_output.npy")
        recorded_training_output = recorded_training_output.tolist()
        interarrival_record=np.load(data_floder + "interarrival_record.npy")
    except Exception as e:
        print(e)
        recorded_training_feature = []
        recorded_training_output = []
        interarrival_record=np.asarray([])
        accumulated_performance = 0
        Q_value_matrix = np.zeros((11,11,11))
        count = 0
        interarrival_data = generate(0.07,100,int(count))
        create_system_interaccival_file(interarrival_data)
        running_server_and_collect_result(5,5,5)
        result = get_mean_latency_from_output_file()
        delete_server_output_files()
        Q_value_matrix[:] = result
        flag_initial_matrix = np.ones((11,11,11))
        print(result)
        #move_and_rename_server_output_files(5,5,5,count)
        recorded_training_feature.append([0.07,5,5,5])
        recorded_training_output.append(result)
        #interarrival_record=np.append(interarrival_record,np.mean(interarrival_data))
        #np.save(data_floder + "recorded_training_feature.npy", recorded_training_feature)
        #np.save(data_floder + "recorded_training_output.npy", recorded_training_output)
        default_performance = 0
        for t in range(5):
            running_server_and_collect_result(0,0,0)
            result = get_mean_latency_from_output_file()
            print(t)
            print(result)
            delete_server_output_files()
            print(default_performance)
            default_performance = (default_performance * t + result) / (t + 1)
            recorded_training_feature.append([0.07,0,0,0])
            recorded_training_output.append(result)
        np.save(data_floder + "default_performance.npy", default_performance)
        assert count==0
        assert interarrival_record.size==0
    print(np.size(recorded_training_output))
    print(np.size(recorded_training_feature))
    

    while count<3000:
        if count<800:
            global_parameter_interarrival_time=0.07
        elif count>=800 and count<1300:
            global_parameter_interarrival_time=0.06
        elif count>=1300 and count<2000:
            global_parameter_interarrival_time=0.07
        elif count>=2000 and count<2400:
            global_parameter_interarrival_time=0.10
        elif count>=2400 and count<3000:
            global_parameter_interarrival_time=0.12
        else:
            print(count)
            raise Exception
        interarrival_data = generate(global_parameter_interarrival_time,100,int(count))
        interarrival_record=np.append(interarrival_record,np.mean(interarrival_data))
        create_system_interaccival_file(interarrival_data)

        try:
            average_accumulated_performance = accumulated_performance / count
        except :
            average_accumulated_performance = 0

        t_matrix = Q_value_matrix.copy()
        #to prevent overflow from something like exp(1000)
        t_matrix=1/t_matrix*1000000*20
        while np.max(t_matrix)>500:
            t_matrix=t_matrix/2
        matrix_exp=np.exp(t_matrix)
        total=matrix_exp.sum()
        rand=random.uniform(0, total)
        flag_continue=True
        for ti in range(matrix_exp.shape[0]):
            for tj in range(matrix_exp.shape[1]):
                for tk in range(matrix_exp.shape[2]):
                    rand=rand-matrix_exp[ti,tj,tk]
                    if rand<=0:
                        flag_continue=False
                        break
                if flag_continue is False:
                    break
            if flag_continue is False:
                break
        print("The chosen configuration is",ti, tj, tk)
        print("The chance of this configuration is",matrix_exp[ti,tj,tk]/total*100,"%")
        print("The best configuration known so far is",np.unravel_index(np.argmax(matrix_exp),matrix_exp.shape))
        print("The chance of the best configuration known so far is",np.amax(matrix_exp)/total*100,"%")


        running_server_and_collect_result(ti, tj, tk)
        result = get_mean_latency_from_output_file()
        recorded_training_feature.append([global_parameter_interarrival_time,ti,tj,tk])
        recorded_training_output.append(result)
        accumulated_performance = accumulated_performance + result
        print("Predicted result is ",Q_value_matrix[ti,tj,tk])
        print("Actual result is ",result)
        print("The accumulated average performance so far is",average_accumulated_performance / 1000,"ms")
        move_and_rename_server_output_files(ti, tj, tk,count)
        count = count + 1
        estimated_Q_matrix=update_value_matrix_with_policy_network(recorded_training_feature, recorded_training_output,global_parameter_interarrival_time)
        for t_i in range(0,11):
            for t_j in range(0,11):
                for t_k in range(0,11):
                    t_a = np.array((ti ,tj, tk))
                    t_b = np.array((t_i ,t_j, t_k))
                    parameter_alpha=0.3
                    if np.linalg.norm(t_a - t_b) <= 3: #the update region size
                        Q_value_matrix[t_i,t_j,t_k] = (1 - parameter_alpha) * Q_value_matrix[t_i,t_j,t_k] + parameter_alpha * estimated_Q_matrix[t_i,t_j,t_k]#the region based update

        np.save(data_floder + "Q-matrix",Q_value_matrix)
        np.save(data_floder + "global_step",count)
        np.save(data_floder + "flag_initial_matrix",flag_initial_matrix)
        np.save(data_floder + "recorded_training_feature.npy", recorded_training_feature)
        np.save(data_floder + "recorded_training_output.npy", recorded_training_output)
        np.save(data_floder + "accumulated_performance.npy", accumulated_performance)
        np.save(data_floder + "interarrival_record.npy", interarrival_record)

def signal_handler(signal, frame):
    print('Detected Ctrl+C, shutting down!')
    sys.exit(0)

if __name__ == "__main__":
    main()

signal.signal(signal.SIGINT, signal_handler)
