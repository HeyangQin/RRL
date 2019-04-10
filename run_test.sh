#!/bin/bash

#if [$# -eq 1]; then
    
#elif

shutdown_server(){
		printf "Killing Tensorflow server ..\n"
		pkill tensorflow_mode
}
shutdown_client(){
		printf "Killing the Inception Client..\n"
		pkill python*
}
start_server(){
	echo "[Test- BS->$batch_size, BT->$batch_threads, INTER_OP->$inter_op, INTRA_OP->$intra_op "
	bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --enable_batching --port=9000 --model_name=inception --model_base_path=your_model_path --batch_queue=1000  --batch_timeout=4000000 --batch_size=$batch_size --batch_threads=$batch_threads  --intra_op=$intra_op --inter_op=$inter_op &>> $TEST_FILE &
	echo "Server is running!!"
}

wait_until_server_ready(){
	while true; do
                NUMBER_OF_LINES=$(tail $SERVER_LOG | grep "Running ModelServer at" | wc -l)
                if [ $NUMBER_OF_LINES -gt 0 ]; then
                        break
                fi
        done
        echo "[I am here  TEST - BS->$batch_size, BT->$batch_threads, INTER_OP->$inter_op, INTRA_OP->$intra_op, TEST_NUMBER->$exec_times]:Server Ready..."
}

client_start(){
  echo "[starting the clien with numbe of images-> $NUMBER_OF_IMAGES ]"

	  ../../serving_1.4/serving/bazel-bin/tensorflow_serving/example/inception_client --number_of_images=$NUMBER_OF_IMAGES --server=localhost:9000 --inter_arrival=0.0 >> $CLIENT_FILE 
	echo "[Test Finished- BS->$batch_size, BT->$batch_threads, INTER_OP->$inter_op, INTRA_OP->$intra_op "	
#echo "Client is running"
}

run_test(){
                 touch $TEST_FILE
                 touch $CLIENT_FILE
                 NUMBER_OF_IMAGES=$((batch_size*500))
                 shutdown_client
                 shutdown_server
                 start_server
                 sleep 30
                 echo "Waiting is finished"
                 client_start

}

run_bt(){
inter_op=1
intra_op=1
for batch_size in {10..30..10} ; do
	mkdir $TEST_FOLDER$batch_size'_'$TIME
        for batch_threads in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 35 40 ; do    
        	TEST_FILE=$HOME'/'$TEST_FOLDER$batch_size'_'$TIME'/bs_'${batch_size}'_bt_'${batch_threads}'_inter_'${inter_op}'_intra_'$intra_op
		CLIENT_FILE=$HOME'/'$TEST_FOLDER$batch_size'_'$TIME'/client_bs_'${batch_size}'_bt_'${batch_threads}'_inter_'${inter_op}'_intra_'$intra_op
               run_test
                
        done
done

}

run_inter_op(){
intra_op=1
batch_threads=1
for batch_size in {10..30..10} ; do
	mkdir $TEST_FOLDER$batch_size'_'$TIME
	for inter_op in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 35 40 ; do
        	TEST_FILE=$HOME'/'$TEST_FOLDER$batch_size'_'$TIME'/bs_'${batch_size}'_bt_'${batch_threads}'_inter_'${inter_op}'_intra_'$intra_op
               CLIENT_FILE=$HOME'/'$TEST_FOLDER$batch_size'_'$TIME'/client_bs_'${batch_size}'_bt_'${batch_threads}'_inter_'${inter_op}'_intra_'$intra_op
		  run_test
	done
done

}

run_intra_op(){
batch_threads=1
inter_op=1
for batch_size in {10..30..10} ; do
	mkdir $TEST_FOLDER$batch_size'_'$TIME
	for intra_op in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 35 40 ; do
                TEST_FILE=$HOME'/'$TEST_FOLDER$batch_size'_'$TIME'/bs_'${batch_size}'_bt_'${batch_threads}'_inter_'${inter_op}'_intra_'$intra_op
		CLIENT_FILE=$HOME'/'$TEST_FOLDER$batch_size'_'$TIME'/client_bs_'${batch_size}'_bt_'${batch_threads}'_inter_'${inter_op}'_intra_'$intra_op
               run_test
        done
done

}

HOME=$(pwd)
TIME=$(date +'%d-%m-%y-%H-%M')
#$MAIN_FOLDER='Test'$TIME
inter_op=1
intra_op=1
batch_threads=1
NUMBER_OF_IMAGES=0
TEST_FOLDER="$1"
a="inter"
echo $TIME
if [ "$TEST_FOLDER" ==  "inter" ]; then
  # echo 'test passed'
     TEST_FOLDER='inter_bs_'
     run_inter_op
elif [ "$TEST_FOLDER" == "intra" ]; then
     TEST_FOLDER='intra_bs_'
     run_intra_op
elif [ "$TEST_FOLDER" == "bt" ]; then
     TEST_FOLDER='bt_bs_'
     run_bt
#elif [$# -eq 0]; then
 #    TEST_FOLDER='test_bs_'
else
     echo "Argunments are not correct"
fi







