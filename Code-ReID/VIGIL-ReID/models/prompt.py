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
        self.n_ctx = 4  
        self.n_cls = num_classes
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype
        
        # 为每个类别创建上下文向量
        ctx_vectors = torch.empty(num_classes, self.n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Meta network
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        # Prompt template
        self.ctx_init = self._get_prompt_template(dataset_name)
        tokenized_prompts = clip.tokenize(self.ctx_init).to(self.device)
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :5])  
        self.register_buffer("token_suffix", embedding[:, 5+self.n_ctx:])

    def forward_stage1(self, pids, image_features):
        """Stage 1: 使用图像特征指导的prompt生成"""
        batch_size = image_features.shape[0]
        
        # 生成图像引导的bias
        bias = self.meta_net(image_features)  # [batch_size, ctx_dim]
        bias = bias.unsqueeze(1)  # [batch_size, 1, ctx_dim]
        
        # 获取对应pid的context
        ctx = self.ctx[pids]  # [batch_size, n_ctx, ctx_dim]
        ctx = ctx + bias  # 添加图像引导的bias
        
        # 扩展prefix和suffix
        prefix = self.token_prefix.expand(batch_size, -1, -1)
        suffix = self.token_suffix.expand(batch_size, -1, -1)
        
        # 组合prompt
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts
    
    def forward_stage2(self):
        """Stage 2: 生成所有类别的prompts"""
        # 使用学习到的context
        ctx = self.ctx  # [n_cls, n_ctx, ctx_dim]
        
        # 扩展prefix和suffix
        prefix = self.token_prefix.expand(self.n_cls, -1, -1)
        suffix = self.token_suffix.expand(self.n_cls, -1, -1)
        
        # 组合prompt
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts
    
    def forward(self, *args, **kwargs):
        """保持向后兼容"""
        return self.forward_stage2()

    def _get_prompt_template(self, dataset_name):
        prompts = {
            "FriesianCattle2017": "a photo of a X X X X cattle.",
            "MPDD": "a photo of a X X X X dog.",
            "ATRW": "a photo of a X X X X tiger.",
        }
        return prompts.get(dataset_name, "a photo of a X X X X.")