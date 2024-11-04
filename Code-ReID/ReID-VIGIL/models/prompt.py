# models/prompt.py

import torch
import torch.nn as nn
from collections import OrderedDict
import clip

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, num_classes, dataset_name, clip_model):
        super().__init__()
        
        self.device = next(clip_model.parameters()).device
        self.n_ctx = 4  # number of context tokens
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype
        
        # Define dataset-specific prompts
        ctx_vectors = torch.empty(num_classes, self.n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Create prompt templates based on dataset
        self.ctx_init = self._get_prompt_template(dataset_name)
        
        # Meta network for feature-guided prompt learning
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        # Tokenize prompts and move to correct device
        tokenized_prompts = clip.tokenize(self.ctx_init).to(self.device)
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
        # Get embeddings
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)
        
        # Split into prefix, suffix and register buffers
        self.register_buffer("token_prefix", embedding[:, :5])  # SOS + "a photo of a"
        self.register_buffer("token_suffix", embedding[:, 5+self.n_ctx:])  # EOS

        

    def forward(self, label, image_features=None):
        batch_size = label.shape[0]
        
        if image_features is not None:
            # Stage 1: Use image-guided context
            bias = self.meta_net(image_features)
            bias = bias.unsqueeze(1)
            ctx = self.ctx[label] + bias
        else:
            # Stage 2: Use basic context
            ctx = self.ctx[label]
            
        # Ensure all tensors are on the same device
        prefix = self.token_prefix.expand(batch_size, -1, -1).to(label.device)
        suffix = self.token_suffix.expand(batch_size, -1, -1).to(label.device)
        ctx = ctx.to(label.device)
        
        # Ensure ctx has the correct shape
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        elif ctx.dim() == 3 and ctx.size(0) == 1:
            ctx = ctx.expand(batch_size, -1, -1)
            
        # Cat the full prompt
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts
    
    def _get_prompt_template(self, dataset_name):
        """Get dataset-specific prompt template."""
        prompts = {
            "FriesianCattle2017": "a photo of a X X X X cattle.",
            "MPDD": "a photo of a X X X X dog.",
            "ATRW": "a photo of a X X X X tiger.",
            # Add other datasets as needed
        }
        return prompts.get(dataset_name, "a photo of a X X X X.")