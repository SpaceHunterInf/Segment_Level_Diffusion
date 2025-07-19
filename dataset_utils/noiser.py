import random
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from contextlib import contextmanager

@contextmanager
def temp_freeze_params(model):
    requires_grad_status = {}
    for name, param in model.named_parameters():
        requires_grad_status[name] = param.requires_grad
        param.requires_grad_(False)
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            param.requires_grad_(requires_grad_status[name])

class GaussianNoiser:
    #TODO a noise scheduler???
    def __init__(self, trigger_chance=0.1, noise_ratio=0.01, seed=None):
        self.trigger_chance = trigger_chance
        self.current_noise_std = noise_ratio

    def __call__(self, latent):
        if random.random() < self.trigger_chance:
            return torch.zeros_like(latent)  # Return original if not triggered
        else:
            noise = torch.randn_like(latent) * self.current_noise_std
            return noise
        
class ADVNoiser:
    def __init__(self, trigger_chance=1, epsilon=0.1, seed=None, td_noise_ratio=0.01):
        self.epsilon = epsilon
        self.trigger_chance = trigger_chance
        self.td_noise_ratio = td_noise_ratio
        
    def r_adv_loss(self, model, org_latent_state, gold_labels):
        # Temporarily freeze model parameters
        with temp_freeze_params(model):
            latent_states = org_latent_state.clone().detach()
            latent_states.requires_grad_(True)
            
            encoder_outputs = BaseModelOutput(last_hidden_state=model.get_decoder_input(latent_states))
            outputs = model(
                encoder_outputs=encoder_outputs,
                labels=gold_labels
            )
            
            outputs.loss.backward()
            
            g = latent_states.grad
            if g is None or torch.all(g == 0):
                r_adv = torch.zeros_like(org_latent_state)
            else:
                norm = torch.norm(g, p=2, dim=(1,2), keepdim=True)
                r_adv = self.epsilon * g / (norm + 1e-8)
            
        output = model(labels=gold_labels, encoder_outputs=BaseModelOutput(last_hidden_state=model.get_decoder_input(org_latent_state + r_adv)))
        adv_loss = output.loss
        
        return adv_loss, r_adv
    
    
    def v_adv_loss(self, model, org_latent_state):
        # Temporarily freeze model parameters
        with temp_freeze_params(model):
            latent_states = org_latent_state.clone().detach()
            
            # Get original probabilities
            with torch.no_grad():
                encoder_outputs = BaseModelOutput(last_hidden_state=model.get_decoder_input(latent_states))
                outputs = model.generate(
                    encoder_outputs=encoder_outputs,
                    max_length=50,  
                    min_length=10, 
                    num_beams=1,    # Use greedy decoding for speed
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    pad_token_id=model.config.pad_token_id,  # Enable padding

                )
                generated_ids = outputs.sequences
                max_length = generated_ids.size(1)
                
            outputs = model(encoder_outputs=encoder_outputs, decoder_input_ids=generated_ids)
            log_probs_original = F.log_softmax(outputs.logits, dim=-1)
            
            # Get perturbed probabilities
            # Step 2: Create noise tensor for perturbation
            d = torch.randn_like(org_latent_state)
            d_norm = torch.norm(d, p=2, dim=-1, keepdim=True)
            d = d / (d_norm + 1e-8) * self.td_noise_ratio
            
            #calculate the gradient w.r.t. s+d 
            perturbed_latent_states = latent_states + d
            perturbed_latent_states.requires_grad_(True)
            
            encoder_outputs = BaseModelOutput(last_hidden_state=model.get_decoder_input(perturbed_latent_states))
            
            with torch.no_grad():
                outputs = model.generate(
                    encoder_outputs=encoder_outputs,
                    max_length=max_length,  # Adjust as needed
                    num_beams=1,    # Use greedy decoding for speed
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    pad_token_id=model.config.pad_token_id,  # Enable padding           
                )
                generated_ids = outputs.sequences
            outputs = model(encoder_outputs=encoder_outputs, decoder_input_ids=generated_ids)
            probs_perturbed = F.softmax(outputs.logits, dim=-1)
            
            kl_div = F.kl_div(log_probs_original, probs_perturbed + 1e-8, reduction='batchmean') #ensuring its numerically stable
            kl_div.backward()
            
            g = perturbed_latent_states.grad
            if g is None or torch.all(g == 0):
                r_v_adv = torch.zeros_like(org_latent_state)
            else:
                norm = torch.norm(g, p=2, dim=(1,2), keepdim=True)
                r_v_adv = self.epsilon * g / (norm + 1e-8)
            
            # print('logprobs', log_probs_original)
            # print('perturbed', probs_perturbed)
            # print('kl_div', kl_div)
            # print('gradient', g, )
            # print('rvadv', r_v_adv)
            # breakpoint()
        
        encoder_outputs = BaseModelOutput(last_hidden_state=model.get_decoder_input(org_latent_state + r_v_adv))
        outputs = model.generate(encoder_outputs=encoder_outputs,
                    max_length=max_length,  # Adjust as needed
                    num_beams=1,    # Use greedy decoding for speed
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    pad_token_id=model.config.pad_token_id,  # Enable padding                     
                    )
                                
        generated_ids = outputs.sequences
        outputs = model(encoder_outputs=encoder_outputs, decoder_input_ids=generated_ids)
        probs = F.softmax(outputs.logits, dim=-1)
        
        #print('Outside')
        #print(log_probs_original.shape, probs.shape)
        v_adv_loss = F.kl_div(log_probs_original, probs + 1e-8, reduction='batchmean')
                
        return v_adv_loss, r_v_adv
    
class SubNoiser:
    def __init__(self, vocab_size, trigger_chance=0.1, sub_prob=0.15, seed=None, special_tokens=None):
        """
        Initialize the SubNoiser with specified parameters.

        Args:
        - vocab_size (int): The size of the tokenizer's vocabulary. Used to generate random token IDs.
        - trigger_chance (float): The probability of applying the noiser to a given input.
        - sub_prob (float): The percentage of token IDs to substitute with random tokens.
        - seed (int, optional): The seed for random number generators to ensure reproducibility.
        - special_tokens (list, optional): A list of special token IDs that should not be substituted or used as substitutes.
        """
        self.vocab_size = vocab_size
        self.trigger_chance = trigger_chance
        self.sub_prob = sub_prob
        self.seed = seed
        self.special_tokens = special_tokens if special_tokens else []

        # Set the seed if provided for reproducibility
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def __call__(self, token_ids):
        """
        Apply the noiser to the input token IDs.

        Args:
        - token_ids (torch.Tensor): A tensor of token IDs to be potentially modified.

        Returns:
        - torch.Tensor: A tensor with some token IDs potentially substituted with random token IDs.
        """
        # Check if we should trigger the noiser
        if random.random() < self.trigger_chance:
            return token_ids  # Return original if not triggered

        # Convert tensor to numpy for easier manipulation
        token_ids_np = token_ids.cpu().numpy()
        total_tokens = len(token_ids_np)

        # Determine the number of substitutions based on sub_prob
        
        num_subs = int(total_tokens * self.sub_prob)
        #print(total_tokens, num_subs)
        if num_subs == 0:
            return token_ids  # No substitutions to make, return original
        
        # Generate a set of valid token IDs for substitution (excluding special tokens)
        valid_token_ids = set(range(self.vocab_size)) - set(self.special_tokens)
        valid_token_ids = list(valid_token_ids)

        # Randomly select positions to substitute (excluding special tokens)
        sub_indices = [idx for idx in range(total_tokens) if token_ids_np[idx] not in self.special_tokens]
        sub_indices = random.sample(sub_indices, min(num_subs, len(sub_indices)))

        # Substitute token IDs at the selected positions with random token IDs (excluding special tokens)
        for idx in sub_indices:
            token_ids_np[idx] = random.choice(valid_token_ids)

        # Convert back to torch tensor and return
        return torch.tensor(token_ids_np, dtype=token_ids.dtype)

# Example usage
if __name__ == "__main__":
    # Load the T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    # Get special token IDs (example: pad_token_id, eos_token_id, bos_token_id)
    special_tokens = [
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id
    ]
    
    # Initialize the noiser with a 10% trigger chance, 15% substitution probability, a seed, and special tokens
    noiser = SubNoiser(vocab_size=tokenizer.vocab_size, trigger_chance=1, sub_prob=0.3, seed=2024, special_tokens=special_tokens)

    # Example input: A sample sentence
    input_text = "This approach ensures that your noise injection does not interfere with the model's understanding of sentence structure or padding behavior, maintaining the integrity of special tokens during training."
    
    # Tokenize the input text
    tokenized_input = tokenizer.encode(input_text, return_tensors='pt')[0]
    
    # Print original token IDs and their corresponding tokens
    print("Original Token IDs:", tokenized_input.tolist())
    print("Original Tokens:   ", tokenizer.convert_ids_to_tokens(tokenized_input.tolist()))

    # Apply the noiser
    noisy_token_ids = noiser(tokenized_input)

    # Print noisy token IDs and their corresponding tokens
    print("Noisy Token IDs:   ", noisy_token_ids.tolist())
    print("Noisy Tokens:      ", tokenizer.convert_ids_to_tokens(noisy_token_ids.tolist()))
