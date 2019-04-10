#!/bin/bash

while true ; do

	free -m | awk 'NR==2{printf "%.2f%%\t %.2f%% \t%.2f%%\n", $3*100/$2,$2,$3}'
	sleep 1

done
