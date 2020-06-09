for scale in 1 0.1 0.01
do
	for latents in 8 16 32 64
	do
		python ml4cvd/recipes.py --mode train --tensors /data/partners_ecg/mgh/ --input_tensors partners_ecg_5000_only_lead_I_zero_mean_std1_scale_$scale --output_tensors partners_ecg_5000_only_lead_I_zero_mean_std1_scale_$scale --training_steps 128 --validation_steps 16 --test_steps 10 --batch_size 64 --bottleneck_type variational --dense_layers $latents --epochs 10 --output_folder /home/nate/ml/train_runs/vae_compare6 --id vae_${latents}_latents_${scale}_scale --cache_size 0 --learning_rate 1e-3 --train_csv /home/nate/ml/train_ids.csv --valid_csv /home/nate/ml/validation_ids.csv --test_csv /home/nate/ml/test_ids.csv
	done
done

