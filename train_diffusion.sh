WANDB_MODE=offline python train_text_diffusion.py --dataset_name full_roc_utt_jsonl --learning_rate 2e-4 \
--num_train_steps 100000 --train_batch_size 16 --eval_batch_size 16 --tx_dim 768 --tx_depth 12 --objective pred_v \
--enc_dec_model google/flan-t5-base --num_samples 32 --self_condition --scale_shift --loss_type l2 \
--train_schedule cosine --wandb_name roc-flan-t5_noised-decoding_loss-post_ae --sampling_timesteps 250 \
--latent_model_path saved_latent_models/your_model_name_here \
--save_and_sample_every 5000 --num_dense_connections 3  --optimizer adamw --train_prob_self_cond 0.5 \
--max_diffusion_len 320 --max_output_seq_length 64 --num_of_sentences 5 \
--parallel_decoding --seq2seq_candidates 3 \
--specified_context_encoder your_local_context_encoder_here_not_hf_repo_link \
--decoding_loss --decoding_loss_weight 0.5 --post_ae_loss --post_ae_loss_weight 0.5 --gradient_accumulation_steps 4