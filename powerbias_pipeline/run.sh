# example of script that executes multiple sequential runs in the background
# run it using:
# ./run.sh &> out &
# in order to save the output in "out" file
# also execute "disown" right after to log out the terminal
nohup mpirun python3 run.py /home/ubuntu/POWERBIAS_outputs/mergers/z0.yaml do_pk --do-bias &
pid=$!
wait $pid
kill -9 $pid
nohup mpirun python3 run.py /home/ubuntu/POWERBIAS_outputs/mergers/z1.yaml do_pk --do-bias &
pid=$!
wait $pid
kill -9 $pid
nohup mpirun python3 run.py /home/ubuntu/POWERBIAS_outputs/mergers/z2.yaml do_pk --do-bias &
pid=$!
wait $pid
kill -9 $pid
nohup mpirun python3 run.py /home/ubuntu/POWERBIAS_outputs/mergers/z3.yaml do_pk --do-bias &
pid=$!
wait $pid
kill -9 $pid
nohup mpirun python3 run.py /home/ubuntu/POWERBIAS_outputs/mergers/z4.yaml do_pk --do-bias &
pid=$!
wait $pid
kill -9 $pid
nohup mpirun python3 run.py /home/ubuntu/POWERBIAS_outputs/mergers/z5.yaml do_pk --do-bias &
pid=$!
wait $pid
kill -9 $pid
nohup mpirun python3 run.py /home/ubuntu/POWERBIAS_outputs/mergers/z6.yaml do_pk --do-bias &
pid=$!
wait $pid
kill -9 $pid
