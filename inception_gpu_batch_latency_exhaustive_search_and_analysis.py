from __future__ import print_function
import time
import subprocess
import os
import sys
import signal
import os.path
import psutil



def delete_server_output_files():
    try:
        os.remove('direct_session.txt')
        os.remove('main.txt')
        os.remove('main_start.txt')
        os.remove('main_end.txt')
        os.remove('direct_session_start.txt')
        os.remove('direct_session_end.txt')
    except:
        pass

delete_server_output_files()

binary_folder = 
def running_server_and_collect_result(intel, intra, number_of_thread):
    FNULL = open(os.devnull, 'w')
    if number_of_thread == 0 or intel == 0:
        model_server_process = subprocess.call([binary_folder + '/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server',
                '--port=9000', '--enable_batching=false', '--model_name=inception',
                '--model_base_path='])#,stdout=FNULL,stderr=FNULL
    else:
        model_server_process = subprocess.call([binary_folder + '/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server',
                '--port=9000', '--enable_batching=true', '--model_name=inception',
                '--model_base_path=',
                '--batch_size=' + str(intel),
                '--batch_timeout=' + str(intra),
                '--batch_threads=' + str(number_of_thread)])#,stdout=FNULL,stderr=FNULL
    print("Model server shut down.")
    return 0
binary_folder=binary_folder
data_floder=
def move_and_rename_server_output_files(i,j,k):
    
    if not os.path.exists(data_floder):
        os.makedirs(data_floder)
    
    os.rename('main.txt',data_floder+"main_"+str(i)+"_"+str(j)+"_"+str(k)+".txt")
    os.rename('direct_session.txt',data_floder+"direct_session_"+str(i)+"_"+str(j)+"_"+str(k)+".txt")
    os.rename('main_start.txt',data_floder+"main_start_"+str(i)+"_"+str(j)+"_"+str(k)+".txt")
    os.rename('main_end.txt',data_floder+"main_end_"+str(i)+"_"+str(j)+"_"+str(k)+".txt")
    os.rename('direct_session_start.txt',data_floder+"direct_session_start_"+str(i)+"_"+str(j)+"_"+str(k)+".txt")
    os.rename('direct_session_end.txt',data_floder+"direct_session_end_"+str(i)+"_"+str(j)+"_"+str(k)+".txt")

def main():
    start_from_i=0;
    start_from_j=0;
    start_from_k=0;
    for ti in range(50,51):
        for tj in range(0,55,5):
            for tk in range(11):
                print(ti, tj, tk)
                if os.path.isfile(data_floder+"main_"+str(ti)+"_"+str(tj)+"_"+str(tk)+".txt"):
                    print("Already done.")
                else:
                    if ti*11*11+tj*11+tk >= start_from_i*11*11+start_from_j*11+start_from_k:
                        running_server_and_collect_result(ti, tj, tk)
                        move_and_rename_server_output_files(ti, tj, tk)
                        print("output files saved.")
                    else:
                        print("skipped.")
    print("All test episodes finished.", end='\r')


def signal_handler(signal, frame):
    print('Detected Ctrl+C, shutting down!')
    sys.exit(0)

if __name__ == "__main__":
    main()

signal.signal(signal.SIGINT, signal_handler)
