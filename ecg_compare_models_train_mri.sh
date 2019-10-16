#!/usr/bin/bash

TENSORS="/mnt/disks/sax-lax-40k/2019-10-01/"
OUTPUT_TENSOR_LIST="ecg_semi_coarse"

for output_tensor in ${OUTPUT_TENSOR_LIST}
do
    echo "Training for "${output_tensor} 
    ./scripts/tf.sh /home/${USER}/ml/ml4cvd/recipes.py --mode train \
		--tensors ${TENSORS} \
		--input_tensors ecg_rest \
		--output_tensors LA_2Ch_vol_max LVM LVESV \
		--training_steps 96 --validation_steps 32 --test_steps 24 \
		--batch_size 32 --epochs 96 --patience 12 \
		--output_folder /home/${USER}/ml/trained_models/ \
		--id ${USER}_ecg_heartchambers --learning_rate 0.00002
done
