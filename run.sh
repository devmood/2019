export DATA="./mydata"
export TRAIN="train.csv"
export TEST="test.csv"
export GT="gt.csv"
export SUBMISSION="submission_popular.csv"

spark-submit preprocess-trainset.py \
--executor-memory 8g \
--executor-cores 8 \
--driver-memory 10g

cp ./mydata/mytrain.csv/*.csv ${DATA}/${TRAIN}
cp ./mydata/mytest.csv/*.csv ${DATA}/${TEST}
cp ./mydata/mygt.csv/*.csv ${DATA}/${GT}

python src/baseline_algorithm/rec_popular.py --data-path $DATA
verify-submission --data-path=$DATA --submission-file $SUBMISSION --test-file $TEST
score-submission --data-path=$DATA --submission-file $SUBMISSION --ground-truth-file $GT
