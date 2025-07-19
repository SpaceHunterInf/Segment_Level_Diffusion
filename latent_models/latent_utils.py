import re, os
from transformers import AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration, AutoModelForCausalLM, MBartTokenizerFast, MT5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration

import CONSTANTS as CONSTANTS

from latent_models.bart_latent_model import BARTForConditionalGenerationLatent
from latent_models.t5_latent_model import T5ForConditionalGenerationLatent, MT5ForConditionalGenerationLatent



def get_latent_model(args):
    if 'bart' in args.enc_dec_model:
        config = BartForConditionalGeneration.from_pretrained(
            args.enc_dec_model).config
        lm = BARTForConditionalGenerationLatent.from_pretrained(
            args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            args.enc_dec_model)
    elif 't5' in args.enc_dec_model:
        if 'mt5' in args.enc_dec_model:
            config = MT5ForConditionalGeneration.from_pretrained(
                args.enc_dec_model).config
            lm = MT5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                args.enc_dec_model)
        else:
            config = T5ForConditionalGeneration.from_pretrained(
                args.enc_dec_model).config
            
            if args.specified_tokenizer != None:
                tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                    args.specified_tokenizer)
            else:
                tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                    args.enc_dec_model)
            
            if args.direct_connection:
                print('Using traditional encoder decoder')
                lm = T5ForConditionalGeneration.from_pretrained(args.enc_dec_model)
            else:
                lm = T5ForConditionalGenerationLatent.from_pretrained(
                    args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers,
                    l2_normalize_latents=args.l2_normalize_latents, _fast_init=False, max_seq_len=args.max_seq_len)   
            
            if args.specified_tokenizer != None:
                lm.resize_token_embeddings(len(tokenizer))
                # if 'utt' in args.dataset_name:
                #     new_tokens = ['[PICK]', 'z[FINISHED]', '<', '[USER_SYS]'] + ['[USER_{}]'.format(str(i)) for i in range(6)]
                # else:
                #     new_tokens = ['[PICK]', '[UTT_BREAK]', '[FINISHED]', '<', '[USER_SYS]'] + ['[USER_{}]'.format(str(i)) for i in range(6)]
                # tokenizer.add_tokens(new_tokens)
                # lm.resize_token_embeddings(len(tokenizer))
                # tokenizer.save_pretrained(os.path.join('datasets', args.dataset_name, 'tokenizer'))

    else:
        print("Unsupported model")
        raise NotImplementedError
    
    if args.lm_mode == 'ft':
        for (param_name, param) in lm.named_parameters():
            param.requires_grad = True
    elif args.lm_mode == 'freeze':
        for (param_name, param) in lm.named_parameters():
            if re.fullmatch(".*perceiver.*", param_name):
                param.requires_grad = True
                print(f"Trainable: {param_name}")
            else:
                param.requires_grad = False
    else:
        raise NotImplementedError

    return lm, tokenizer, config