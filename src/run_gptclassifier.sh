 #!/bin/bash 

TEMPERATURE=0
NUM_COMPLETIONS=1
RESULTS_DIR=../results/

for LANGUAGE in nepali colombian
do
    for num_examples in 10
    do
        TEST_DATA_FILE=../data/${LANGUAGE}_test.csv
        TRAIN_DATA_FILE=../data/incontext_${LANGUAGE}.csv

        python driver.py \
        --test_data_file ${TEST_DATA_FILE} \
        --train_data_file ${TRAIN_DATA_FILE} \
        --results_dir ${RESULTS_DIR} \
        --language ${LANGUAGE} \
        --temperature ${TEMPERATURE} \
        --num_completions ${NUM_COMPLETIONS} \
        --num_examples ${num_examples} \
        --reflect \
        --use_content_summary \
        --use_chain_of_thought
    done
done