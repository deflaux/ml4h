#!/usr/bin/bash

TENSORS="/mnt/disks/ecg-rest-31k/2019-06-10/"
OUTPUT_TENSOR_LIST="ecg_rhythm ecg_coarse ecg_semi_coarse ecg_semi_coarse_with_poor \
		    ecg_normal ecg_infarct ecg_block acute_mi anterior_blocks av_block\ 
                    fine_rhythms incomplete_right_bundle_branch_block infarcts \
                    left_atrial_enlargement left_ventricular_hypertrophy lvh_fine 
                    prolonged_qt p_axis p_duration" 

for output_tensor in ${OUTPUT_TENSOR_LIST}
do
    ./scripts/tf.sh /home/${USER}/ml/ml4cvd/recipes.py --mode train \
		--tensors ${TENSORS} \
		--input_tensors ecg_rest \
		--output_tensors ${output_tensor} \
		--training_steps 96 --validation_steps 32 --test_steps 24 \
		--batch_size 32 --epochs 64 --patience 12 \
		--output_folder /home/${USER}/ml/trained_models/ \
		--id ${USER}_${INPUT_TENSORS}_${output_tensor} --learning_rate 0.00002
done
