import math
import copy
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
import csv
import timeit
import json
import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import wandb
from latent_models.bart_latent_model import BARTForConditionalGenerationLatent
from latent_models.t5_latent_model import T5ForConditionalGenerationLatent

import diffusion.constant as constant
import diffusion.optimizer as optimizer
import dataset_utils.sentence_dataset as sentence_dataset
from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from latent_models.latent_utils import get_latent_model
from evaluation import evaluation
from diffusion.gaussian_diffusion import *

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_v'])

class Trainer(object):
    def __init__(
        self,
        args,
        diffusion,
        dataset_name,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        save_and_sample_every = 5000,
        num_samples = 25,
        seq2seq_candidates = 10,
        seq2seq_train_context_encoder = False,
        results_folder = './results',
        amp = False,
        mixed_precision = 'no'
    ):
        super().__init__()


        set_seeds(42)

        self.args = args

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=90))

        self.accelerator = Accelerator(
            mixed_precision = mixed_precision,
            log_with='wandb',
            kwargs_handlers=[ddp_kwargs, init_process_kwargs]
        )
        self.num_devices = self.accelerator.num_processes
        args.num_devices = self.num_devices

        if self.accelerator.is_main_process:
            if args.output_dir is None:
                args.output_dir = file_utils.get_output_dir(args)
                with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                    json.dump(args.__dict__, f, indent=2)
            results_folder = args.output_dir
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        

        self.diffusion = diffusion
        
        self.post_ae_loss = self.args.post_ae_loss
        self.post_ae_loss_weight = self.args.post_ae_loss_weight
        
        self.decoding_loss = self.args.decoding_loss
        self.decoding_loss_weight = self.args.decoding_loss_weight

        self.num_samples = num_samples
        self.seq2seq_candidates = seq2seq_candidates
        self.save_and_sample_every = save_and_sample_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.max_seq_len = diffusion.max_seq_len

        self.latent_model_path = args.latent_model_path

        self.enc_dec_model_path = args.enc_dec_model
        
        self.parallel_encoding = args.parallel_encoding
        self.parallel_decoding = args.parallel_decoding

        # Init Encoder-decoder model
        if 'bart' in args.enc_dec_model:
            self.enc_dec_model = BartForConditionalGeneration.from_pretrained(self.enc_dec_model_path)
        elif 'flan-t5' in args.enc_dec_model:
            self.enc_dec_model = T5ForConditionalGeneration.from_pretrained(self.enc_dec_model_path, torch_dtype=torch.bfloat16)
        elif 'mt5' in args.enc_dec_model:
            self.enc_dec_model = MT5ForConditionalGeneration.from_pretrained(self.enc_dec_model_path, torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f'invalid enc_dec_model {args.enc_dec_model}')
        
        data_path = os.path.join('datasets', dataset_name)

        self.diffusion.using_latent_model = False
        self.seq2seq = self.diffusion.diffusion_model.seq2seq
        self.class_conditional = self.diffusion.diffusion_model.class_conditional
        self.seq2seq_unconditional_prob = self.diffusion.seq2seq_unconditional_prob
        self.best_seq2seq_metric = 0
        self.context_tokenizer = None
        if args.latent_model_path:
            device = self.accelerator.device
            with open(os.path.join(args.latent_model_path, 'args.json'), 'rt') as f:
                latent_model_args = json.load(f)
            
            latent_argparse = argparse.Namespace(**latent_model_args)
            if args.specified_context_encoder == None:
                self.diffusion.context_encoder = self.enc_dec_model.get_encoder()
            else:
                print('loading specified context encoder from ' + args.specified_context_encoder)
                print(os.listdir(args.specified_context_encoder))
                self.diffusion.context_encoder = T5ForConditionalGeneration.from_pretrained(args.specified_context_encoder).get_encoder()
            
            self.seq2seq_train_context_encoder = seq2seq_train_context_encoder
            if seq2seq_train_context_encoder:
                for param in self.diffusion.context_encoder.parameters():
                    param.requires_grad = True
            else:
                for param in self.diffusion.context_encoder.parameters():
                    param.requires_grad = False

            self.enc_dec_model, self.tokenizer, _ = get_latent_model(latent_argparse)
            if args.specified_tokenizer != None:
                print('Reloading tokenizer at ' + os.path.join(args.latent_model_path, 'tokenizer'))
                self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.latent_model_path, 'tokenizer'))
                print(len(self.tokenizer))
                self.enc_dec_model.resize_token_embeddings(len(self.tokenizer))
            
            self.context_tokenizer = self.tokenizer
            data = torch.load(os.path.join(args.latent_model_path, 'model.pt'), map_location=device)
            if type(data) == dict and 'model' in data.keys():
                self.enc_dec_model.load_state_dict(data['model'])
            else:
                self.enc_dec_model.load_state_dict(data)
            
            self.num_of_sentences = args.num_of_sentences
            self.diffusion.max_seq_len = latent_argparse.num_encoder_latents * self.num_of_sentences
            
            self.num_encoder_latents = latent_argparse.num_encoder_latents
            self.num_decoder_latents = latent_argparse.num_decoder_latents
            self.diffusion.using_latent_model = True
            self.direct_connection = latent_argparse.direct_connection
            self.diffusion.l2_normalize = (hasattr(self.enc_dec_model, 'l2_normalize_latents') and self.enc_dec_model.l2_normalize_latents)
            if self.diffusion.l2_normalize:
                assert not args.normalize_latent
            for param in self.enc_dec_model.parameters():
                param.requires_grad = False
        self.using_latent_model = self.diffusion.using_latent_model
        
        self.enc_dec_model.eval()
        self.mse_loss = torch.nn.MSELoss()
        # dataset and dataloader
        self.dataset_name = dataset_name
        self.dataloader, self.val_dataloader, self.test_dataloader, self.train_val_dataloader = sentence_dataset.prepare_data(data_path, self.tokenizer,
                                                                                                             train_batch_size=self.train_batch_size,
                                                                                                             val_batch_size=self.eval_batch_size,
                                                                                                             max_input_seq_length=self.args.max_input_seq_length,
                                                                                                             max_output_seq_length=self.args.max_output_seq_length,
                                                                                                             parallel_encoding=self.parallel_encoding,
                                                                                                             parallel_decoding=self.parallel_decoding,
                                                                                                             max_input_sentences=self.num_of_sentences,
                                                                                                             max_output_sentences=self.num_of_sentences)
        # Subsample train and val splits for computing language generation during runtime
        
        #breakpoint()
        if not self.seq2seq:
            training_lengths = [min(sum(self.dataloader.dataset[idx]['attention_mask']), self.max_seq_len) for idx in range(self.dataloader.dataset.num_rows)]
            length_counts = Counter(training_lengths)
            probs = torch.tensor([length_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.max_seq_len+1)])
            assert probs[0] == 0, 'Can\'t have examples of length 0'
            self.length_categorical = torch.distributions.Categorical(probs=probs)

        if self.class_conditional:
            training_labels = [self.dataloader.dataset[idx]['label'] for idx in range(self.dataloader.dataset.num_rows)]
            label_counts = Counter(training_labels)
            probs = torch.tensor([label_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.diffusion.diffusion_model.num_classes)])
            self.class_categorical = torch.distributions.Categorical(probs=probs)
        
        
        if self.decoding_loss:
                        # Get the parameters of the T5 decoder
            reconstruction_params = self.enc_dec_model.perceiver_ae.perceiver_decoder.parameters()
            t5_decoder_params = self.enc_dec_model.get_decoder().parameters()
            self.combined_decoder_params = list(reconstruction_params) + list(t5_decoder_params)
            for param in self.combined_decoder_params:
                param.requires_grad = True

            print(f"Diffusion trainable parameters: {sum(p.numel() for p in diffusion.parameters())}")
            print(f"Decoder trainable_parameters: {sum(p.numel() for p in self.combined_decoder_params)}")
            # Optimizer
            self.opt = optimizer.get_adamw_optimizer(diffusion.parameters(), lr=train_lr, betas=adam_betas, weight_decay=adam_weight_decay)
            self.dec_opt = optimizer.get_adamw_optimizer(self.combined_decoder_params, lr=1e-5, betas=adam_betas, weight_decay=adam_weight_decay) #TODO refine this
            # Scheduler
            lr_scheduler = get_scheduler(
                lr_schedule,
                optimizer=self.opt,
                num_warmup_steps=num_warmup_steps*self.num_devices,
                num_training_steps=train_num_steps*self.num_devices,
            )
            
            dec_lr_scheduler = get_scheduler(
                lr_schedule,
                optimizer=self.dec_opt,
                num_warmup_steps=num_warmup_steps*self.num_devices,
                num_training_steps=train_num_steps*self.num_devices,
            )
        else:
            
            num_parameters = sum(p.numel() for p in diffusion.parameters())
            print(f"Total number of trainable parameters: {num_parameters}")

            # optimizer

            self.opt = optimizer.get_adamw_optimizer(diffusion.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)

            # scheduler

            lr_scheduler = get_scheduler(
                lr_schedule,
                optimizer=self.opt,
                num_warmup_steps=num_warmup_steps*self.num_devices,
                num_training_steps=train_num_steps*self.num_devices,
            )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion, beta = ema_decay, update_every = ema_update_every, power=3/4)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        if self.decoding_loss:
            self.diffusion, self.enc_dec_model, self.opt, self.dataloader, self.lr_scheduler, self.dec_opt, self.dec_lr_scheduler = self.accelerator.prepare(self.diffusion, self.enc_dec_model, self.opt, self.dataloader, lr_scheduler, self.dec_opt, dec_lr_scheduler)
        else:
            self.diffusion, self.enc_dec_model, self.opt, self.dataloader, self.lr_scheduler = self.accelerator.prepare(self.diffusion, self.enc_dec_model, self.opt, self.dataloader, lr_scheduler)
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)
        self.reference_dict = {}

    def save(self, best=False):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'scheduler': self.lr_scheduler.state_dict(),
        }
        if best:
            torch.save(data, str(self.results_folder / f'best_model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model.pt'))
            
        if self.decoding_loss and best:
            self.enc_dec_model.save_pretrained(os.path.join(self.results_folder, 'best_enc_dec_model'))
            self.tokenizer.save_pretrained(os.path.join(self.results_folder, 'best_enc_dec_model'))
        elif self.decoding_loss:
            self.enc_dec_model.save_pretrained(os.path.join(self.results_folder, 'enc_dec_model'))
            self.tokenizer.save_pretrained(os.path.join(self.results_folder, 'enc_dec_model'))
        else:
            pass

    def load(self, file_path=None, best=False, init_only=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        if best:
            data = torch.load(str(file_path / f'best_model.pt'), map_location=device)
        else:
            data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.diffusion)
        # For backwards compatibility with earlier models
        model.load_state_dict(data['model'])
        
        # if self.decoding_loss:
        #                 # Adjust the optimizer state dict to match the new parameter groups
        #     saved_opt_state = data['opt']
        #     saved_opt_state['param_groups'] = self.opt.state_dict()['param_groups']

        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_local_main_process:
            self.ema.load_state_dict(data['ema'])
        if init_only:
            return
        self.step = data['step']
        
        if 'scheduler' in data:
            self.lr_scheduler.load_state_dict(data['scheduler'])
        # For backwards compatibility with earlier models
        
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def load_enc_dec(self, args):
        
        file_path = args.decoding_reload_folder
        
        if 'bart' in args.enc_dec_model:
            tmp_enc_dec_model = BartForConditionalGeneration.from_pretrained(file_path)
        elif 'flan-t5' in args.enc_dec_model:
            tmp_enc_dec_model = T5ForConditionalGeneration.from_pretrained(file_path)
        elif 'mt5' in args.enc_dec_model:
            tmp_enc_dec_model = MT5ForConditionalGeneration.from_pretrained(file_path)
        else:
            raise ValueError(f'invalid enc_dec_model {file_path}')
        tmp_state_dict = tmp_enc_dec_model.state_dict()
        self_state_dict = self.enc_dec_model.state_dict()

        # Filter out keys that exist in both models and match in shape
        matching_params = {k: v for k, v in tmp_state_dict.items() if k in self_state_dict and self_state_dict[k].shape == v.shape}

        # Update self.enc_dec_model's state_dict
        self_state_dict.update(matching_params)

        # Load the updated state_dict back into self.enc_dec_model
        self.enc_dec_model.load_state_dict(self_state_dict)

        print('Reload enc_dec_model from ' + file_path)

    def log_reference_metrics(self, test=False):
        accelerator = self.accelerator
        if test:
            train_subset = self.dataset['train']['text'][:self.num_samples]
            train_subset2 = self.dataset['train']['text'][self.num_samples:(2*self.num_samples)] 
            test_subset = self.dataset['test']['text'][:self.num_samples]
            self.reference_dict['reference/test_perplexity'] = evaluation.compute_perplexity(test_subset)
            for mauve_model_id in ["gpt2-large"]:
                self.reference_dict[f'reference/{mauve_model_id}_train_test_mauve'], _ = evaluation.compute_mauve(train_subset, test_subset, mauve_model_id)
                self.reference_dict[f'reference/{mauve_model_id}_train_train_mauve'], _ = evaluation.compute_mauve(train_subset, train_subset2, mauve_model_id)
                ngram_metrics = evaluation.compute_diversity(test_subset)
            for k, v in ngram_metrics.items():
                self.reference_dict[f"reference/test_{k}"] = v
            self.reference_dict[f"reference/test_memorization"] = evaluation.compute_memorization(test_subset, self.dataset['train']['text'])
            self.reference_dict['reference/test_unique_wordcount'] = evaluation.compute_wordcount(test_subset)
            return

        val_subset = self.dataset['valid']['text'][:self.num_samples]
        train_subset = self.dataset['train']['text'][:self.num_samples]
        train_subset2 = self.dataset['train']['text'][self.num_samples:(2*self.num_samples)] 
        self.reference_dict['reference/train_perplexity'] = evaluation.compute_perplexity(train_subset)
        self.reference_dict['reference/val_perplexity'] = evaluation.compute_perplexity(val_subset)
        for mauve_model_id in ["gpt2-large"]:
            self.reference_dict[f'reference/{mauve_model_id}_train_val_mauve'], _ = evaluation.compute_mauve(train_subset, val_subset, mauve_model_id)
            self.reference_dict[f'reference/{mauve_model_id}_train_train_mauve'], _ = evaluation.compute_mauve(train_subset, train_subset2, mauve_model_id)
        ngram_metrics = evaluation.compute_diversity(val_subset)
        for k, v in ngram_metrics.items():
            self.reference_dict[f"reference/val_{k}"] = v
        ngram_metrics = evaluation.compute_diversity(train_subset)
        for k, v in ngram_metrics.items():
            self.reference_dict[f"reference/train_{k}"] = v
        self.reference_dict[f"reference/val_memorization"] = evaluation.compute_memorization(val_subset, self.dataset['train']['text'])
        self.reference_dict['reference/train_unique_wordcount'] = evaluation.compute_wordcount(train_subset)
        self.reference_dict['reference/val_unique_wordcounts'] = evaluation.compute_wordcount(val_subset)
        torch.cuda.empty_cache() 
        
    @torch.no_grad()
    def sample(self, num_samples=None, class_id=None, seed=42, test=False, cls_free_guidance=1.0):
        num_samples = default(num_samples, self.num_samples)
        accelerator = self.accelerator
        device = accelerator.device
        self.diffusion.to('cpu')
        torch.cuda.empty_cache() 

        self.ema.ema_model.eval()

        # Extract references
        reference_texts = {}
        if exists(class_id):
            for filter_class_id in range(self.diffusion.diffusion_model.num_classes):
                filtered_dataset = self.dataset.filter(lambda example: example["label"]==filter_class_id)
                if test:
                    reference_texts[f'ref{filter_class_id}_test'] = filtered_dataset['test']['text']
                    continue
                reference_texts[f'ref{filter_class_id}_val'] = filtered_dataset['valid']['text']
                reference_texts[f'ref{filter_class_id}_train'] = filtered_dataset['train']['text']
            
            for key, reference_text in reference_texts.items():
                num_samples = min(num_samples, len(reference_text))
            reference_texts = {k: v[:num_samples] for k, v in reference_texts.items()}
        else:
            if test:
                reference_texts[f'test'] = self.dataset['test']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]
            else:
                reference_texts['val'] = self.dataset['valid']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]

        milestone = self.step // self.save_and_sample_every
        # Stores generation outputs for each strategy
        all_texts_lists = {k:[] for k,_ in constant.generate_kwargs.items()}    

        torch.manual_seed(seed)
        def get_class_id(n):
            if exists(class_id):
                return torch.tensor([class_id]*n, dtype=torch.long, device=device)
            if self.class_conditional:
                if self.diffusion.diffusion_model.class_unconditional_prob > 0:
                    return torch.tensor([self.diffusion.diffusion_model.num_classes]*n, dtype=torch.long, device=device)
                return self.class_categorical.sample((n,)).to(device)
            return None
        # Loop until enough senetences have been generated across all strategies 
        while min([len(all_texts_lists[ele]) for ele in all_texts_lists]) < num_samples:
            batches = num_to_groups(num_samples-min([len(all_texts_lists[ele]) for ele in all_texts_lists]), max(self.eval_batch_size,self.train_batch_size))
            model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.ema.ema_model.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=get_class_id(n), cls_free_guidance=cls_free_guidance)), batches))
            
            for (latents, mask) in model_outputs:
                latents, mask = latents.to(device), mask.to(device)
                if self.args.normalize_latent:
                    latents = self.ema.ema_model.unnormalize_latent(latents)
                for k, kwargs in constant.generate_kwargs.items():
                    if self.latent_model_path:
                        attention_mask = None
                        encoder_output = BaseModelOutput(last_hidden_state=self.enc_dec_model.get_decoder_input(latents.clone()))
                    else:
                        attention_mask = mask.clone()
                        encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                    sample_ids = self.enc_dec_model.generate(encoder_outputs=encoder_output, attention_mask=attention_mask, **kwargs)
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                    all_texts_lists[k].extend(texts_list)
        
        assert min([len(all_texts_lists[ele]) for ele in all_texts_lists]) >= num_samples
        text_generations = {k:v[:num_samples] for k,v in all_texts_lists.items()} 

        metrics = {}

        self.ema.to('cpu')
        torch.cuda.empty_cache() 
        for strategy, all_texts_list in text_generations.items():
            class_id_prefix = f'cond{class_id}_' if exists(class_id) else ''
            file_utils.save_text_samples(all_texts_list, os.path.join(self.results_folder, f'{"eval-" if self.args.eval else ""}{f"eval{seed}-" if self.args.eval_test else ""}{class_id_prefix}{strategy}-sample-{milestone}.txt'))
            metrics[f"model/{strategy}/{class_id_prefix}perplexity"] = evaluation.compute_perplexity(all_texts_list)
            metrics[f"model/{strategy}/{class_id_prefix}unique_wordcount"] = evaluation.compute_wordcount(all_texts_list)
            ngram_metrics = evaluation.compute_diversity(all_texts_list)
            for k, v in ngram_metrics.items():
                metrics[f"model/{strategy}/{class_id_prefix}{k}"] = v
            metrics[f"model/{strategy}/{class_id_prefix}memorization"] = evaluation.compute_memorization(all_texts_list, self.dataset['train']['text'])
            table = wandb.Table( 
                columns=['Samples'], data=[[text] for text in all_texts_list])
            accelerator.log({f"model/{strategy}/{class_id_prefix}samples": table}, self.step)

            # Only evaluate MAUVE if generations are reasonable to speed up validation early on
            if metrics[f"model/{strategy}/{class_id_prefix}perplexity"] > 5000:
                continue

            for mauve_model_id in ["gpt2-large"]:
                for key, reference_text in reference_texts.items():
                    metrics[f"model/{strategy}/{mauve_model_id}_{class_id_prefix}{key}_mauve"], _ = evaluation.compute_mauve(all_texts_list, reference_text, mauve_model_id)

        if len(self.reference_dict) == 0 or test:
            self.log_reference_metrics(test)
        if test:
            metrics_dict = {**metrics,**self.reference_dict}
            metrics_dict = {f'{k}_seed{seed}':v for k,v in metrics_dict.items()}
            accelerator.log(metrics_dict, self.step)
            print(metrics_dict)
        else:
            accelerator.log({**metrics,**self.reference_dict}, self.step)
        torch.cuda.empty_cache() 
        self.diffusion.to(device)
        self.ema.to(device)

    @torch.no_grad()
    def sample_seq2seq(self, num_samples=None, split='val', seed=42, num_candidates=None, cls_free_guidance=1.0,):
        assert split in ['train', 'val', 'test']
        num_samples = default(num_samples, self.num_samples) if split != 'test' else len(self.test_dataloader.dataset)
        num_candidates = default(num_candidates, self.seq2seq_candidates)
        accelerator = self.accelerator
        device = accelerator.device

        self.ema.ema_model.eval()

        # Extract references
        reference_texts = []
        source_texts = []
        pred_texts = []

        torch.manual_seed(seed)

        if split == 'val':
            dataloader = self.val_dataloader
            prefix = ''
        elif split == 'train':
            dataloader = self.train_val_dataloader
            prefix = 'train/'
        elif split == 'test':
            dataloader = self.test_dataloader
            prefix = 'test/'
        else:
            raise ValueError(f'invalid split {split}')
        
        diffusion = accelerator.unwrap_model(self.diffusion)
        prefix += f'guide{cls_free_guidance}/' if cls_free_guidance != 1.0 else ''
        for batch in dataloader:
            data = batch
            current_batch_size = data['input_ids'].shape[0]
            
            #reshape input for batch encoding, as outputs needs to be disentangled
            output_ids = data['input_ids'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
            output_attention_mask = data['attention_masks'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
            #input doesn't need to be disentangled, at least for now
            if self.parallel_encoding:
                cond_input_ids = data['cond_input_ids'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                cond_attention_mask = data['cond_attention_masks'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                seq2seq_cond = diffusion.context_encoder(input_ids = cond_input_ids, attention_mask = cond_attention_mask).last_hidden_state.float()
                seq2seq_mask = cond_attention_mask.bool()
                
                single_seq_length = seq2seq_cond.shape[1]
                            
                seq2seq_cond = seq2seq_cond.reshape(current_batch_size, single_seq_length * self.args.num_of_sentences, -1)
                seq2seq_mask = seq2seq_mask.reshape(current_batch_size, single_seq_length * self.args.num_of_sentences)
            else:
                cond_input_ids =  data['cond_input_ids'].view(current_batch_size, -1).to(device)
                cond_attention_mask = data['cond_attention_masks'].view(current_batch_size, -1).to(device)
                seq2seq_cond = diffusion.context_encoder(input_ids = cond_input_ids, attention_mask = cond_attention_mask).last_hidden_state.float()
                seq2seq_mask = cond_attention_mask.bool()
            
            pred_cand_list = []
            ref_cand_list = []
            source_cand_list = []
            gen_kwargs = constant.generate_kwargs['beam']
            gen_kwargs['max_length'] = self.args.max_output_seq_length
            
            for _ in range(num_candidates):
                l2_normalize = (hasattr(self.enc_dec_model, 'l2_normalize_latents') and self.enc_dec_model.l2_normalize_latents)
                latents, mask = self.ema.ema_model.sample(batch_size=seq2seq_cond.shape[0], length=None, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
                
                #reshape latent to de-concate sentences
                #TODO you might not want to assume self.num_encoder_latents == self.num_decoder_latents
                latents = latents.reshape(current_batch_size, self.num_encoder_latents * self.args.num_of_sentences, -1)
                latents = latents.reshape(current_batch_size * self.args.num_of_sentences, self.num_encoder_latents, -1)
                mask = mask.reshape(current_batch_size, self.num_encoder_latents * self.args.num_of_sentences, -1)
                mask = mask.reshape(current_batch_size * self.args.num_of_sentences, self.num_encoder_latents, -1)
                
                if self.args.normalize_latent:
                    latents = self.ema.ema_model.unnormalize_latent(latents)
                if self.latent_model_path and not(self.direct_connection):
                    attention_mask = None
                    encoder_output = BaseModelOutput(last_hidden_state=self.enc_dec_model.get_decoder_input(latents.clone()))
                else:
                    if self.direct_connection:
                        attention_mask = None
                    else:
                        attention_mask = mask.clone()
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                    
                #reshape the output into single sentences
                sample_ids = self.enc_dec_model.generate(encoder_outputs=encoder_output, attention_mask=attention_mask, **gen_kwargs)
                
                texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in sample_ids]
                refs_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in output_ids]
                sources_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in cond_input_ids]
                new_texts_list = []
                new_refs_list = []
                new_sources_list = []
                
                #breakpoint()
                if self.parallel_decoding:
                    for sentence_idx in range(0, len(texts_list),self.args.num_of_sentences):
                        start_idx = sentence_idx
                        new_texts_list.append(' [UTT_BREAK] '.join(texts_list[start_idx:start_idx+self.args.num_of_sentences]))
                        new_refs_list.append(' [UTT_BREAK] '.join(refs_list[start_idx:start_idx+self.args.num_of_sentences]))
                    
                    pred_cand_list.append(new_texts_list)
                    ref_cand_list.append(new_refs_list)
                    
                if self.parallel_encoding:
                    for sentence_idx in range(0, len(sources_list),self.args.num_of_sentences):
                        start_idx = sentence_idx
                        new_sources_list.append(' [UTT_BREAK] '.join(sources_list[start_idx:start_idx+self.args.num_of_sentences]))
                else: 
                    new_sources_list = sources_list
                source_cand_list.append(new_sources_list)
                
            assert len(pred_cand_list) == num_candidates
            assert len(ref_cand_list) == num_candidates
            assert len(source_cand_list) == num_candidates
            pred_texts.extend([val for tup in zip(*pred_cand_list) for val in tup])
            reference_texts.extend([val for tup in zip(*ref_cand_list) for val in tup])
            source_texts.extend([val for tup in zip(*source_cand_list) for val in tup])
            
            output_path = 'results/result.jsonl'  # Updated to use relative path

            with open(output_path, 'w') as f:
                for pred, ref, src in zip(pred_texts, reference_texts, source_texts):
                    # Write each trio as a JSON-like dictionary
                    line = {
                        "prediction": pred,
                        "reference": ref,
                        "source": src
                    }
                    f.write(f"{line}\n")

            print(f"Data written {print(len(pred_texts))}")
            
            if len(pred_texts) >= num_samples*num_candidates:
                break
            
        assert len(pred_texts) == len(reference_texts) == len(source_texts)
        assert len(pred_texts) >= num_samples*num_candidates
        pred_texts = pred_texts[:num_samples*num_candidates]
        reference_texts = reference_texts[:num_samples*num_candidates]
        source_texts = source_texts[:num_samples*num_candidates]

        # Save samples and references to json
        #breakpoint()
        if split == 'test':
            #breakpoint()
            samples_dict = {'pred_texts': pred_texts, 'reference_texts': reference_texts, 'source_texts': source_texts}
            save_path = os.path.join(self.results_folder, f'{prefix}_seq2seq_{split}_samples.json')    
            # Create dir if it doesn't exist   
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            with open(os.path.join(save_path), 'w') as f:
                json.dump(samples_dict, f)

        # Log samples
        # source | reference | pred
        columns = ['source', 'reference', 'pred']
        data = []
        for i in range(len(reference_texts)):
            row = [source_texts[i], reference_texts[i], pred_texts[i]]
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        accelerator.log({f"seq2seq/{prefix}{split}_samples": table}, self.step)

        # Compute metrics
        metrics = {}

        if 'wmt' in self.dataset_name:
            tokenize = 'intl' if self.dataset_name == 'wmt14-en-de' else '13a'

            if num_candidates > 1:
                mbr_sacrebleu_scores = np.zeros((num_samples, num_candidates))
                for i in range(num_candidates):
                    pred_texts_i = pred_texts[i::num_candidates]
                    for j in range(num_candidates):
                        if j == i:
                            continue
                        ref_texts_j = pred_texts[j::num_candidates]
                        sacrebleu_arr = np.array([evaluation.compute_sacrebleu([pred], [ref], tokenize=tokenize, use_effective_order=True) for pred, ref in zip(pred_texts_i, ref_texts_j)])
                        mbr_sacrebleu_scores[:, i] += sacrebleu_arr
                best_indices = np.argmax(mbr_sacrebleu_scores, axis=1)
                best_predictions = [pred_texts[i*num_candidates + idx] for i, idx in enumerate(best_indices)]
                if split == 'test':
                    gt_reference_texts = self.dataset['test']['text'][:num_samples]
                elif split == 'val':
                    gt_reference_texts = self.dataset['valid']['text'][:num_samples]
                elif split == 'train':
                    gt_reference_texts = reference_texts[::num_candidates]
                else:
                    raise NotImplementedError
                metrics[f'model/seq2seq/{prefix}mbr_sacrebleu'] = evaluation.compute_sacrebleu(best_predictions, gt_reference_texts, tokenize=tokenize)
        else:
            # Get oracle rouge
            raw_rouge_metrics = evaluation.compute_rouge(pred_texts, reference_texts, use_aggregator=False)
            # Compute the max rouge score across num_candidates
            for k, v in raw_rouge_metrics.items():
                np_metric = np.array(v).reshape(num_samples, num_candidates)
                np_metric = np.max(np_metric, axis=1)
                metrics[f"model/seq2seq/{prefix}oracle_{k}"] = np_metric.mean().item()

            if num_candidates > 1:
                mbr_rouge_scores = np.zeros((num_samples, num_candidates))
                for i in range(num_candidates):
                    pred_texts_i = pred_texts[i::num_candidates]
                    for j in range(num_candidates):
                        if j == i:
                            continue
                        ref_texts_j = pred_texts[j::num_candidates]
                        rouge2_arr = np.array(evaluation.compute_rouge(pred_texts_i, ref_texts_j, use_aggregator=False)['rouge2'])
                        mbr_rouge_scores[:, i] += rouge2_arr
                best_indices = np.argmax(mbr_rouge_scores, axis=1)
                best_predictions = [pred_texts[i*num_candidates + idx] for i, idx in enumerate(best_indices)]
                mbr_rouge_metrics = evaluation.compute_rouge(best_predictions, reference_texts[::num_candidates])
                for k, v in mbr_rouge_metrics.items():
                    metrics[f"model/seq2seq/{prefix}mbr_{k}"] = v
                metrics[f'model/seq2seq/{prefix}mbr_bertscore'] = evaluation.compute_bertscore(best_predictions, reference_texts[::num_candidates])

        # Get every num_candidates samples
        pred_texts = pred_texts[::num_candidates]
        reference_texts = reference_texts[::num_candidates]
        source_texts = source_texts[::num_candidates]
        
        if 'wmt' in self.dataset_name:
            save_path = os.path.join(self.results_folder, f'{prefix}{split}_samples.txt')   
            # Create dir if it doesn't exist
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            file_utils.save_text_samples(pred_texts, save_path)
            tokenize = 'intl' if self.dataset_name == 'wmt14-en-de' else '13a'
            # Compute BLEU
            if split == 'test':
                assert num_samples == len(self.dataset['test']['text'])
                reference_texts = self.dataset['test']['text'][:num_samples]
            elif split == 'val':
                reference_texts = self.dataset['valid']['text'][:num_samples]
            assert len(pred_texts) == len(reference_texts)
            sacrebleu_score = evaluation.compute_sacrebleu(pred_texts, reference_texts, tokenize=tokenize)
            metrics[f"model/seq2seq/{prefix}sacrebleu"] = sacrebleu_score
            if metrics[f'model/seq2seq/{prefix}sacrebleu'] > self.best_seq2seq_metric and split == 'val' and cls_free_guidance == 1.0:
                self.best_seq2seq_metric = metrics[f'model/seq2seq/{prefix}sacrebleu']
                self.save(best=True)
        else:
            rouge_metrics = evaluation.compute_rouge(pred_texts, reference_texts)
            for k, v in rouge_metrics.items():
                metrics[f"model/seq2seq/{prefix}{k}"] = v

            if rouge_metrics['rougeL'] > self.best_seq2seq_metric and split == 'val':
                self.best_seq2seq_metric = rouge_metrics['rougeL']
                self.save(best=True)

            rouge_metrics = evaluation.compute_rouge(pred_texts, reference_texts, use_stemmer=True)
            for k, v in rouge_metrics.items():
                metrics[f"model/seq2seq/{prefix}stem_{k}"] = v

            shuffled_pred_texts = random.sample(pred_texts, len(pred_texts))
            shuffled_rouge_metrics = evaluation.compute_rouge(shuffled_pred_texts, reference_texts)
            for k, v in shuffled_rouge_metrics.items():
                metrics[f"model/seq2seq/{prefix}shuffled_{k}"] = v

            #metrics[f"model/seq2seq/{prefix}perplexity"] = evaluation.compute_perplexity(pred_texts)
            metrics[f"model/seq2seq/{prefix}unique_wordcount"] = evaluation.compute_wordcount(pred_texts)
            ngram_metrics = evaluation.compute_diversity(pred_texts)
            for k, v in ngram_metrics.items():
                metrics[f"model/seq2seq/{prefix}{k}"] = v
            #metrics[f"model/seq2seq/{prefix}memorization"] = evaluation.compute_memorization(pred_texts, self.dataset['train']['text'])
            metrics[f"model/seq2seq/{prefix}bertscore"] = evaluation.compute_bertscore(pred_texts, reference_texts)
        
        accelerator.log(metrics, self.step)
        print(metrics)
        torch.cuda.empty_cache() 

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                #TODO center and normalize BART latent space with empirical est. of mean/var.

                total_loss = 0.
                if self.decoding_loss:
                    total_decoding_loss = 0.
                if self.post_ae_loss:
                    total_post_ae_loss = 0.

                for grad_accum_step in range(self.gradient_accumulate_every):
                    data = next(self.data_iter)
                    current_batch_size = data['input_ids'].shape[0]
                    
                    #reshape input for batch encoding, as outputs needs to be disentangled
                    output_ids = data['input_ids'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                    output_attention_mask = data['attention_masks'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                    #input doesn't need to be disentangled, at least for now
                    
                    if self.parallel_encoding:
                        cond_input_ids = data['cond_input_ids'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                        cond_attention_mask = data['cond_attention_masks'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                    else:
                        cond_input_ids =  data['cond_input_ids'].view(current_batch_size, -1).to(device)
                        cond_attention_mask = data['cond_attention_masks'].view(current_batch_size, -1).to(device)
                    
                    with torch.no_grad():
                        #breakpoint()
                        encoder_outputs = self.enc_dec_model.get_encoder()(input_ids = output_ids, attention_mask = output_attention_mask)
                        #last hidden state shape (batch*sentences * max_sentence_length * hidden_dim)
                        
                        #breakpoint()
                        if self.using_latent_model and not(self.direct_connection):
                            latent = self.enc_dec_model.get_diffusion_latent(encoder_outputs, output_attention_mask)      
                        else:                      
                            latent = encoder_outputs.last_hidden_state

                        #reshape latents to concatenate sentences
                        latent = latent.reshape(current_batch_size, self.enc_dec_model.num_encoder_latents*self.args.num_of_sentences, -1)
                        
                        #breakpoint()
                        if self.args.normalize_latent:
                            if self.step==0 and grad_accum_step==0:
                                if self.using_latent_model:
                                    latent_vecs = rearrange(latent, 'b s d -> (b s) d')
                                else:
                                    latent_vecs = torch.cat([latent[i][:torch.sum(output_attention_mask[i])] for i in range(latent.shape[0])], dim=0)
                                
                                # Add mean stats to model and EMA wrapper
                                self.diffusion.latent_mean = torch.mean(latent_vecs, dim=0)
                                self.ema.ema_model.latent_mean = self.diffusion.latent_mean

                                # Add var stats to model and EMA wrapper
                                self.diffusion.latent_scale = torch.std(latent_vecs-self.diffusion.latent_mean, unbiased=False)

                                self.ema.ema_model.latent_scale = self.diffusion.latent_scale
                            latent = self.diffusion.normalize_latent(latent)
                        
                    seq2seq_cond = None
                    seq2seq_mask = None
                    with accelerator.autocast():
                        if self.seq2seq and random.random() < (1-self.seq2seq_unconditional_prob):
                            if self.num_devices > 1:
                                seq2seq_cond = self.diffusion.module.context_encoder(input_ids = cond_input_ids, attention_mask = cond_attention_mask).last_hidden_state.float()
                            else:
                                seq2seq_cond = self.diffusion.context_encoder(input_ids = cond_input_ids, attention_mask = cond_attention_mask).last_hidden_state.float()
                            seq2seq_mask = cond_attention_mask.bool()
                            
                            single_seq_length = seq2seq_cond.shape[1]
                            
                            if self.parallel_encoding:
                                seq2seq_cond = seq2seq_cond.reshape(current_batch_size, single_seq_length * self.args.num_of_sentences, -1)
                                seq2seq_mask = seq2seq_mask.reshape(current_batch_size, single_seq_length * self.args.num_of_sentences)

                    if self.using_latent_model:
                        if self.parallel_decoding:
                            mask = torch.ones(latent.shape[0], self.num_encoder_latents * self.args.num_of_sentences, dtype=torch.bool).to(device)
                        else:
                            raise NotImplementedError
                    else:
                        mask = output_attention_mask.bool()

                    if self.decoding_loss or self.post_ae_loss:
                        loss, x_start = self.diffusion(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, return_x_start=True)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        
                        # torch.autograd.set_detect_anomaly(True)
                        x_start = x_start.reshape(current_batch_size * self.args.num_of_sentences, self.num_decoder_latents, -1).detach().clone()
                        label_ids = data['label_ids'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                        reconstructed = self.enc_dec_model.get_decoder_input(x_start)

                        if self.post_ae_loss:
                            gold_post_ae = self.enc_dec_model.get_decoder_input(latent.reshape(current_batch_size * self.args.num_of_sentences, self.num_decoder_latents, -1)).detach().clone()
                            post_ae_loss = self.mse_loss(reconstructed, gold_post_ae) / self.gradient_accumulate_every
                            
                            total_post_ae_loss += post_ae_loss.item()
                            #WTF was I doing with this total loss?
                            total_loss += post_ae_loss.item() * self.post_ae_loss_weight
                            loss += post_ae_loss.item() * self.post_ae_loss_weight
                            
                        if self.decoding_loss: #old train decoder
                            decoding_loss = self.enc_dec_model(encoder_outputs=BaseModelOutput(last_hidden_state=reconstructed), labels=label_ids).loss
                            decoding_loss = decoding_loss / self.gradient_accumulate_every
                            
                            total_decoding_loss += decoding_loss.item()
                            total_loss += decoding_loss.item() * self.decoding_loss_weight
                            loss += decoding_loss.item() * self.decoding_loss_weight
                            
                        
                        self.accelerator.backward(loss)#, retain_graph=True)     
                        
                        accelerator.clip_grad_norm_(self.diffusion.parameters(), self.args.clip_grad_norm)
                        grad_norm = compute_grad_norm(self.diffusion.parameters())
                        accelerator.wait_for_everyone()
                        self.opt.step()
                        self.lr_scheduler.step()
                        self.opt.zero_grad()
                        accelerator.wait_for_everyone()
                            
                            #TODO separate decoding loss and train_decoder
                            
                            # if self.decoding_loss: #old train decoder
                            #     decoding_loss = self.enc_dec_model(encoder_outputs=BaseModelOutput(last_hidden_state=reconstructed), labels=label_ids).loss
                            #     decoding_loss = decoding_loss / self.gradient_accumulate_every
                                
                            #     total_decoding_loss += decoding_loss.item()
                            #     self.accelerator.backward(decoding_loss)
                                
                            #     accelerator.clip_grad_norm_(self.combined_decoder_params, self.args.clip_grad_norm)
                            #     grad_norm = compute_grad_norm(self.combined_decoder_params)
                            #     accelerator.wait_for_everyone()
                            #     self.dec_opt.step()
                            #     self.dec_lr_scheduler.step()
                            #     self.dec_opt.zero_grad()
                            #     accelerator.wait_for_everyone()
                    else:
                        # breakpoint()
                        # print(latent.shape, mask.shape, seq2seq_cond.shape, seq2seq_mask.shape)
                        loss = self.diffusion(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        self.accelerator.backward(loss)

                        accelerator.clip_grad_norm_(self.diffusion.parameters(), self.args.clip_grad_norm)
                        grad_norm = compute_grad_norm(self.diffusion.parameters())
                        accelerator.wait_for_everyone()
                        self.opt.step()
                        self.lr_scheduler.step()
                        self.opt.zero_grad()
                        accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    logs = {
                        "loss": total_loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                        "step": self.step, 
                        "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), 
                        "samples": self.step*self.train_batch_size*self.gradient_accumulate_every*self.num_devices
                    }
                    if self.post_ae_loss:
                        logs['post_ae_loss'] = total_post_ae_loss
                    if self.decoding_loss:
                        logs['decoding_loss'] = total_decoding_loss
                    self.ema.to(device)
                    self.ema.update()

                    # Log to WandB
                    if self.step % 50 == 0:
                        self.diffusion.eval()
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            total_val_decoding_loss = 0.
                            total_val_post_ae_loss = 0.
                            total_val_ema_loss = 0.
                            total_val_ema_post_ae_loss = 0.
                            total_val_ema_decoding_loss = 0.
                            for grad_accum_step in range(self.gradient_accumulate_every):
                                data = next(self.val_iter)
                                current_batch_size = data['input_ids'].shape[0]
                                
                                #reshape input for batch encoding, as outputs needs to be disentangled
                                output_ids = data['input_ids'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                                output_attention_mask = data['attention_masks'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                                #input doesn't need to be disentangled, at least for now
                                if self.parallel_encoding:
                                    cond_input_ids = data['cond_input_ids'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                                    cond_attention_mask = data['cond_attention_masks'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                                else:
                                    cond_input_ids =  data['cond_input_ids'].view(current_batch_size, -1).to(device)
                                    cond_attention_mask = data['cond_attention_masks'].view(current_batch_size, -1).to(device)
                                
                                encoder_outputs = self.enc_dec_model.get_encoder()(input_ids = output_ids, attention_mask = output_attention_mask)
                                
                                if self.using_latent_model and (not self.direct_connection):
                                    latent = self.enc_dec_model.get_diffusion_latent(encoder_outputs, output_attention_mask)      
                                else:                      
                                    latent = encoder_outputs.last_hidden_state
                                
                                if self.args.normalize_latent:
                                    latent = self.diffusion.normalize_latent(latent)
                                
                                #reshape latent, concatenate sentences
                                latent = latent.reshape(current_batch_size, self.enc_dec_model.num_encoder_latents*self.args.num_of_sentences, -1)
                                
                                seq2seq_cond = None
                                seq2seq_mask = None
                                if self.seq2seq and random.random() < (1-self.seq2seq_unconditional_prob):
                                    with torch.no_grad():
                                        if self.num_devices > 1:
                                            seq2seq_cond = self.diffusion.module.context_encoder(input_ids = cond_input_ids, attention_mask = cond_attention_mask).last_hidden_state.float()
                                        else:
                                            seq2seq_cond = self.diffusion.context_encoder(input_ids = cond_input_ids, attention_mask = cond_attention_mask).last_hidden_state.float()
                                    seq2seq_mask = cond_attention_mask.bool()

                                    single_seq_length = seq2seq_cond.shape[1]
                                    
                                    if self.parallel_encoding:
                                        seq2seq_cond = seq2seq_cond.reshape(current_batch_size, single_seq_length * self.args.num_of_sentences, -1)
                                        seq2seq_mask = seq2seq_mask.reshape(current_batch_size, single_seq_length * self.args.num_of_sentences)                                        
                                            
                                if self.using_latent_model:
                                    if self.parallel_decoding:
                                        mask = torch.ones(latent.shape[0], self.num_encoder_latents * self.args.num_of_sentences, dtype=torch.bool).to(device)
                                else:
                                    mask = output_attention_mask.bool()
                                    
                                    
                                if self.decoding_loss or self.post_ae_loss:
                                    loss, x_start = self.diffusion(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, return_x_start=True)
                                    x_start = x_start.reshape(current_batch_size * self.args.num_of_sentences, self.num_decoder_latents, -1).detach().clone()
                                    label_ids = data['label_ids'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                                    reconstructed = self.enc_dec_model.get_decoder_input(x_start)
                                    
                                    loss = loss / self.gradient_accumulate_every
                                    total_val_loss += loss.item()
                                    logs['val_loss'] = total_val_loss
                                    
                                    if self.post_ae_loss:
                                        gold_post_ae = self.enc_dec_model.get_decoder_input(latent.reshape(current_batch_size * self.args.num_of_sentences, self.num_decoder_latents, -1)).detach().clone()
                                        post_ae_loss = self.mse_loss(reconstructed, gold_post_ae) / self.gradient_accumulate_every
                                        total_val_post_ae_loss += post_ae_loss.item()
                                        logs['val_post_ae_loss'] = total_val_post_ae_loss
                                        
                                    if self.decoding_loss:
                                        decoding_loss = self.enc_dec_model(encoder_outputs=BaseModelOutput(last_hidden_state=reconstructed), labels=label_ids).loss
                                        decoding_loss = decoding_loss / self.gradient_accumulate_every
                                        total_val_decoding_loss += decoding_loss.item()
                                        logs['val_decoding_loss'] = total_val_decoding_loss
                                    
                                    loss, x_start = self.ema.ema_model(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, return_x_start=True)
                                    label_ids = data['label_ids'].view(current_batch_size * self.args.num_of_sentences, -1).to(device)
                                    x_start = x_start.reshape(current_batch_size * self.args.num_of_sentences, self.num_decoder_latents, -1).detach().clone()
                                    reconstructed = self.enc_dec_model.get_decoder_input(x_start)
                                    
                                    loss = loss / self.gradient_accumulate_every
                                    total_val_ema_loss += loss.item()
                                    logs['val_ema_loss'] = total_val_loss
                                    
                                    if self.post_ae_loss:
                                        gold_post_ae = self.enc_dec_model.get_decoder_input(latent.reshape(current_batch_size * self.args.num_of_sentences, self.num_decoder_latents, -1)).detach().clone()
                                        post_ae_loss = self.mse_loss(reconstructed, gold_post_ae) / self.gradient_accumulate_every
                                        total_val_post_ae_loss += post_ae_loss.item()
                                        logs['val_ema_post_ae_loss'] = total_val_post_ae_loss
                                    
                                    if self.decoding_loss:
                                        decoding_loss = self.enc_dec_model(encoder_outputs=BaseModelOutput(last_hidden_state=reconstructed), labels=label_ids).loss
                                        decoding_loss = decoding_loss / self.gradient_accumulate_every
                                        total_val_ema_decoding_loss += decoding_loss.item()
                                        logs['val_ema_decoding_loss'] = total_val_ema_decoding_loss
                                        
                                    #TODO again, need to separate the loss when training decoder
                                    
                                    
                                else:
                                    loss = self.diffusion(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                                    loss = loss / self.gradient_accumulate_every
                                    total_val_loss += loss.item()
                                    loss = self.ema.ema_model(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                                    loss = loss / self.gradient_accumulate_every
                                    total_val_ema_loss += loss.item()
                                    
                                    logs["val_loss"] = total_val_loss
                                    logs["val_ema_loss"] = total_val_ema_loss


                            pbar.set_postfix(**logs)  
                        self.diffusion.train()
                    accelerator.log(logs, step=self.step)
                    if self.step % self.save_and_sample_every == 0:
                        if self.seq2seq:
                            if 'wmt' in self.args.dataset_name:
                                for guidance_strength in [1.0, 2.0]:
                                    self.sample_seq2seq(cls_free_guidance=guidance_strength, incremental=False)
                            else:
                                self.sample_seq2seq()
                            self.sample_seq2seq(split='train')
                        else:
                            self.sample()
                        if self.class_conditional:
                            for class_id in range(self.diffusion.diffusion_model.num_classes):
                                self.sample(num_samples=100, class_id=class_id)
                        self.save()
                        
                        self.diffusion.train() 
                pbar.update(1)
            accelerator.wait_for_everyone()
        self.save()
        accelerator.print('training complete')