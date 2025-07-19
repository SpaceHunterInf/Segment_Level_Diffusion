WANDB_MODE=offline python train_latent_model.py --train_path datasets/example_utt/train.jsonl \
--dev_path datasets/example_utt/valid.jsonl \
--enc_dec_model google/flan-t5-base \
--num_train_epochs 2 --train_batch_size 64 --eval_batch_size 64 --gradient_accumulation_steps 1 \
--num_encoder_latents 64 --num_decoder_latents 64 --dim_ae 64 --num_layers 3 --l2_normalize_latent \
--subnoiser_trigger 0.1 --subnoiser_ratio 0.01 --hnoiser_trigger 0.1 --hnoiser_ratio 0.05 --lm_mode ft --lr 1e-4 --lr_schedule cosine \
--noise_type adv --cse --cse_loss_weight 0.2 \
--wandb_name ecqa_latent_64_64_cnt \
--early_stopping_patience 5 \
--save_dir /home/xz479/rds/hpc-work/Segment_Level_Diffusion/saved_latent_models/test \
