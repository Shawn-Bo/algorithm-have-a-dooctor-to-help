cd ./computes/
python job0_gen_data.py
echo "Job Finished: job0_gen_data.py"
python job1_train.py
echo "Job Finished: job1_train.py"
