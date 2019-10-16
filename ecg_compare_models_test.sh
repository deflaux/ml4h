#!/usr/bin/bash

TENSORS="/mnt/disks/ecg-rest-31k/2019-06-10/"
OUTPUT_TENSOR_LIST="qt_qtc_rr"

for output_tensor in ${OUTPUT_TENSOR_LIST}
do
    echo "Training for "${output_tensor} 
    ./scripts/tf.sh /home/${USER}/ml/ml4cvd/recipes.py --mode compare \
		    --tensors ${TENSORS} \
		    --model_files /home/${USER}/ml/trained_models/${USER}__ecg_semi_coarse/${USER}__ecg_semi_coarse.hd5 \
		    /mnt/ml4cvd/projects/jamesp/data/models/ecg_rest_semi_coarse_only.hd5 \
		    --input_tensors ecg_rest \
		    --output_tensors ecg_semi_coarse \
		    --training_steps 96 --validation_steps 32 --test_steps 24 \
		    --batch_size 32 --epochs 64 --patience 12 \
		    --output_folder /home/${USER}/ml/tested_models/ \
		    --learning_rate 0.00002
done
