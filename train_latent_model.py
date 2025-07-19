import torch
import json, os
import wandb
from transformers import AutoTokenizer, T5ForConditionalGeneration, AdamW, get_scheduler
from transformers.modeling_outputs import BaseModelOutput

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from latent_models.latent_utils import get_latent_model
from utils.simcse import *

import argparse
from evaluation.evaluation import *
from dataset_utils.ae_dataset import *
from dataset_utils.noiser import SubNoiser, GaussianNoiser, ADVNoiser

from datetime import datetime

import torch
import torch.nn.functional as F

#TODO implement T5 VAE
generate_kwargs = {#'beam': {'max_length':64, 'do_sample':False, 'num_beams':4, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2},
                   #'nucleus': {'max_length':64, 'do_sample':True, 'top_p':.95, 'num_beams':1, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2},
                   #'contrastive': {'max_length':64, 'penalty_alpha':0.6, 'top_k':4},
                   'greedy': {'max_length':64}}


class LLM_AE(pl.LightningModule):
    def __init__(self, args, enc_dec_model, decoder_tokenizer:AutoTokenizer, train_steps=None, ckpt_path=None):
        super().__init__()
        self.args = args
        self.lm = enc_dec_model
        self.tokenizer = decoder_tokenizer
        self.num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        self.validation_step_loss = []
        self.validation_step_bleu = []
        self.validation_step_samples = {}
        for strategy in generate_kwargs:
            self.validation_step_samples[strategy] = []
        self.train_steps = train_steps
        self.ckpt_path = ckpt_path
        print(f"Initial learning rate: {self.args.lr}")
        self.noise_type = None
        self.hnoiser = GaussianNoiser(self.args.hnoiser_trigger, self.args.hnoiser_ratio, seed=self.args.seed)
        if self.args.noise_type != None:
            if 'adv' in self.args.noise_type:
                self.noise_type = self.args.noise_type
                self.noiser = ADVNoiser(trigger_chance=1, epsilon=self.args.epsilon, seed=self.args.seed, td_noise_ratio=self.args.td_noise_ratio)
            else:
                raise NotImplementedError
        
    def forward(self, x, compute_adv=True):
        
        encoder_outputs = self.lm.get_encoder()(input_ids = x['encoder_input_ids'], attention_mask = x['attention_mask'])
        diffusion_latent = self.lm.get_diffusion_latent(encoder_outputs, x['attention_mask'])
        
        #add noise here
        loss_dict = {}
        
        noise = self.hnoiser(diffusion_latent)
        diffusion_latent += noise
        
        total_loss = 0.
        encoder_outputs.last_hidden_state = self.lm.get_decoder_input(diffusion_latent)
        output = self.lm(labels=x['label_ids'], encoder_outputs=encoder_outputs)
        lm_loss = output.loss
        loss_dict['lm_loss'] = lm_loss
        total_loss += lm_loss
            
        #adversarial loss
        if compute_adv:
            if self.noise_type == 'r_adv':
                adv_loss, r_adv_noise = self.noiser.r_adv_loss(self.lm, diffusion_latent, x['label_ids'])
                loss_dict['adv_loss'] = adv_loss
                
                total_loss += self.args.adv_loss_weight * adv_loss
            elif self.noise_type == 'v_adv':
                v_adv_loss, v_adv_noise = self.noiser.v_adv_loss(self.lm, diffusion_latent)
                loss_dict['v_adv_loss'] = v_adv_loss
                
                total_loss += self.args.v_adv_loss_weight * v_adv_loss
            elif self.noise_type == 'rv_adv':
                adv_loss, r_adv_noise = self.noiser.r_adv_loss(self.lm, diffusion_latent, x['label_ids'])
                loss_dict['adv_loss'] = adv_loss
                v_adv_loss, v_adv_noise = self.noiser.v_adv_loss(self.lm, diffusion_latent)
                loss_dict['v_adv_loss'] = v_adv_loss
                total_loss += self.args.adv_loss_weight * adv_loss + self.args.v_adv_loss_weight * v_adv_loss
        
        if self.args.cse:
            similar = self.lm.get_encoder()(input_ids = x['similar_ids'], attention_mask = x['similar_attention_mask'])
            similar_diffusion_latent = self.lm.get_diffusion_latent(similar, x['similar_attention_mask'])
            
            contrastive = self.lm.get_encoder()(input_ids = x['contrastive_ids'], attention_mask = x['contrastive_attention_mask'])
            contrastive_diffusion_latent = self.lm.get_diffusion_latent(contrastive, x['contrastive_attention_mask'])
            
            cse_loss = get_simcse_loss(diffusion_latent, similar_diffusion_latent, contrastive_diffusion_latent)
            loss_dict['cse_loss'] = cse_loss
            total_loss += self.args.cse_loss_weight * cse_loss
        
        loss_dict['loss'] = total_loss
        
        return total_loss, loss_dict, encoder_outputs
    
    def training_step(self, batch, batch_idx):
        #self.lm.train()
        loss, loss_dict, _ = self.forward(batch)
        
        for k in loss_dict.keys():
            if k == 'loss':
                self.log('train_loss', loss_dict[k], on_step=True, on_epoch=True, logger=True, prog_bar=True)
            else:
                self.log(f'train_{k}', loss_dict[k], on_step=True, on_epoch=True, logger=True, prog_bar=True)

        # Log learning rate
        optimizer = self.optimizers()
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            self.log("learning_rate", current_lr, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        
        # Log gradient norms
        grad_norm = torch.nn.utils.clip_grad_norm_(self.lm.parameters(), max_norm=1.0)
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        #self.lm.eval()

        loss, loss_dict, encoder_outputs = self.forward(batch, compute_adv=False) # Because it will OOM!

        for k in loss_dict.keys():
            self.log('val_' + k, loss_dict[k])

        self.validation_step_loss.append(loss)
        
        for strategy in generate_kwargs.keys():
            gen_kwargs = generate_kwargs[strategy]
            sample_ids = self.lm.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
            decoded_text = self.tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
            
            try:
                bleu = compute_bleu(decoded_text, batch['decoded_decoder_texts'])
            except:
                bleu = 0
                print(decoded_text, batch['decoded_decoder_texts'])
            
            for i in range(len(batch['decoded_decoder_texts'])):
                item = {'predicted':decoded_text[i], 'gold':batch['decoded_decoder_texts'][i], 'bleu':bleu}
                self.validation_step_samples[strategy].append(item)
            
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_loss).mean()
        for strategy in generate_kwargs.keys():
            bleus = [torch.tensor(x['bleu'], dtype=torch.float) for x in self.validation_step_samples[strategy]]
            bleu_epoch_average = torch.stack(bleus).mean()
            self.log(f'{strategy} bleu', bleu_epoch_average)
        
        self.log("validation_epoch_average", epoch_average)

        columns = ['reference'] + [f'{strategy}/autoencoder' for strategy in generate_kwargs.keys()]
        rows = []
        for i in range(len(self.validation_step_samples[list(generate_kwargs.keys())[0]])):
            row = [self.validation_step_samples[list(generate_kwargs.keys())[0]][i]['gold']]
            for strategy in generate_kwargs.keys():
                row.append(self.validation_step_samples[strategy][i]['predicted'])

            rows.append(row)
        
        self._trainer.logger.log_table(key="val_samples", columns=columns, data=rows)
        
        #breakpoint()
        self.validation_step_loss.clear()
        for k in self.validation_step_samples.keys():
            self.validation_step_samples[k].clear()
        
        self.log("val_loss", epoch_average.item())
        print('val_loss:{}'.format(str(epoch_average.item())))
        torch.save(self.lm.state_dict(), os.path.join(self.ckpt_path, f'epoch{str(self.current_epoch)}-step{str(self.global_step)}-model.pth'))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.lr, correct_bias=True)
        
        num_training_steps = self.trainer.estimated_stepping_batches
        print(f'Total estimated training steps {num_training_steps}')
        num_warmup_steps = min(self.args.lr_warmup_steps * self.num_devices, num_training_steps)

        scheduler = get_scheduler(
            self.args.lr_schedule,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Change to 'epoch' if that's your intended setup
                'frequency': 1
            }
        }


def train(args, *more):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")
    model_name = (args.enc_dec_model + '_' + args.lm_mode + '_' + str(args.seed) + f'subnoiser_{str(args.subnoiser_trigger)}_{str(args.subnoiser_ratio)}' 
                 + '_' + f'hnoiser_{str(args.hnoiser_trigger)}_{str(args.hnoiser_ratio)}' 
                 + '_' + formatted_time)
    
    if args.cse:
        model_name += '_cse'
        
    if args.noise_type !=  None:
        model_name += '_' + args.noise_type
    # train!
    seed_everything(args.seed)
    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_name)

    #For some reason, T5 tokenizer does not have '<'
    lm, tokenizer, config = get_latent_model(args)
    
    if args.resume_training:
        # Construct the path to the checkpoint file
        checkpoint_path = f"{args.resume_dir}/model.pt"
        
        # Load the state dict from the checkpoint file
        checkpoint = torch.load(checkpoint_path)
        
        # Load the state dict into the model
        lm.load_state_dict(checkpoint)
        print(f"Model state loaded from {checkpoint_path}")

    #save model path
    save_path = os.path.join(args.save_dir, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Convert parsed arguments to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a file in JSON format
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args.gradient_accumulation_steps,
                    gradient_clip_val=args.max_norm,
                    max_epochs = args.num_train_epochs,
                    callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.00,
                                                          patience=args.early_stopping_patience,
                                                          verbose=False, mode='min')],
                    devices=args.gpu,
                    deterministic=True,
                    num_nodes=1,
                    #precision=16,
                    accelerator="gpu",
                    logger = wandb_logger
                    )

    # Get special token IDs (example: pad_token_id, eos_token_id, bos_token_id)
    special_tokens = [
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id
    ]

    # Initialize the noiser with a 10% trigger chance, 15% substitution probability, a seed, and special tokens
    noiser = SubNoiser(vocab_size=tokenizer.vocab_size, trigger_chance=args.subnoiser_trigger, sub_prob=args.subnoiser_ratio, seed=args.seed, special_tokens=special_tokens)
    
    train_loader = get_utt_dataloader(args.train_path, tokenizer, tokenizer, max_length=args.max_seq_len, batch_size=args.train_batch_size, noiser=noiser, dev_mode=args.dev_mode, cse=args.cse)
    dev_loader = get_utt_dataloader(args.dev_path, tokenizer, tokenizer, max_length=args.max_seq_len, batch_size=args.eval_batch_size, noiser=noiser, dev_mode=args.dev_mode, cse=args.cse)
    task = LLM_AE(args, lm, decoder_tokenizer=tokenizer, train_steps=len(train_loader), ckpt_path=save_path)
    
    trainer.fit(task, train_loader, dev_loader)
    
    if not args.dev_mode:
        task.tokenizer.save_pretrained(os.path.join(save_path, 'tokenizer'))
        torch.save(task.lm.state_dict(), os.path.join(save_path, 'model.pt'))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=64)
    
    parser.add_argument("--subnoiser_trigger", type=float, default=0.5)
    parser.add_argument("--subnoiser_ratio", type=float, default=0.03)
    
    parser.add_argument("--noise_type", default=None, help='Default noise type for latent representations.')
    parser.add_argument("--hnoiser_trigger", type=float, default=0.5)
    parser.add_argument("--hnoiser_ratio", type=float, default=0.01)
    parser.add_argument("--adv_loss_weight", type=float, default=0.15)
    parser.add_argument("--v_adv_loss_weight", type=float, default=0.15)
    parser.add_argument("--epsilon", type=float, default=0.05, help='Epsilon for adversarial loss norm')
    parser.add_argument("--td_noise_ratio", type=float, default=0.01, help='Ratio of small TD Noise')
    
    parser.add_argument("--enc_dec_model", type=str, default="google/flan-t5-base")
    parser.add_argument("--specified_tokenizer", type=str, default=None)
    
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_encoder_latents", type=int, default=32)
    parser.add_argument("--num_decoder_latents", type=int, default=32)
    parser.add_argument("--dim_ae", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--l2_normalize_latents", action="store_true")
    
    parser.add_argument("--save_dir", type=str, default="saved_latent_models")
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--dev_mode", action='store_true', default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    
    # parser.add_argument("--num_train_steps", type=int, default=None)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    # parser.add_argument("--optimizer", type=str, default="adamw")
    # parser.add_argument("--adam_beta1", type=float, default=0.9)
    # parser.add_argument("--adam_beta2", type=float, default=0.999)
    # parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    # parser.add_argument("--eval_every", type=int, default=1000)
    # parser.add_argument(
    #     "--mixed_precision",
    #     type=str,
    #     default="no",
    #     choices=["no", "fp16", "bf16"],
    #     help=(
    #         "Whether to use mixed precision. Choose"
    #         "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
    #         "and an Nvidia Ampere GPU."
    #     ),
    # )
    parser.add_argument("--wandb_project", type=str, default="t5-vae")
    parser.add_argument("--wandb_name", type=str, default="test")
    parser.add_argument(
        "--lm_mode",
        type=str,
        default="freeze",
        choices=["freeze", "ft",],
        help=(
            "How to fine-tune LM."
        ),
    )
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--direct_connection", action='store_true', default=False)
    parser.add_argument("--cse", action="store_true", default=False)
    parser.add_argument("--cse_loss_weight", type=float, default=0.3)

    parser.add_argument("--early_stopping_patience", type=int, default=-1, help="Number of validation epochs before early stopping, -1 means off")
    args = parser.parse_args()

    train(args)