import os
import argparse
import requests
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import clip # pip install git+https://github.com/openai/CLIP.git
import torch
import random
import math
import wandb
import PIL
from torch import nn
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
from diffusers.loaders import AttnProcsLayers
#from fastprogress import progress_bar, master_bar
from collections import deque
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from diffusers.models.attention_processor import LoRAAttnProcessor
from packaging import version
from transformers import AutoProcessor, AutoModel

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

torch.backends.cuda.matmul.allow_tf32 = True

class pickscore_RM:
    def __init__(self, device = torch.cuda.current_device()):
        
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(self.device)
        self.preprocess = self.diff_preprocess()

    def diff_preprocess(self):

        return Compose([Resize(224, interpolation=BICUBIC, antialias=True),
                CenterCrop(224), self.rescale,
                Normalize((0.48145466,0.4578275,0.40821073), (0.26862954,0.26130258,0.27577711))])

    def rescale(self, image):
        return image*0.00392156862745098

    def get_reward(self, imgs, prompt):
        self.model.to(torch.cuda.current_device())
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(torch.cuda.current_device())

        image_inputs = {}
        image_inputs['pixel_values'] = torch.stack([self.preprocess(img.permute(2, 0, 1)) for img in imgs]).to(torch.cuda.current_device()).float()

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        self.model.to(torch.cuda.current_device())
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        return scores


class aesthetic_RM:
    def __init__(self, device = torch.cuda.current_device()):

        self.device = device
        self.clip_model, _ = clip.load("ViT-L/14", device=self.device)
        self.preprocess = self.diff_preprocess()
        self.aesthetic_model = MLP(768).to(device=self.device)
        self.aesthetic_model.load_state_dict(load_aesthetic_model_weights())

    def rescale(self, image):
        return image/255

    def diff_preprocess(self):

        return Compose([
            Resize(224, interpolation=BICUBIC, antialias=True), CenterCrop(224), self.rescale, 
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    
    def normalize(self, a, dim=-1, p=2):

        l2 = torch.norm(a, p, dim=dim, keepdim=True)

        return a / l2

    def get_reward(self, imgs, *args): 

        diff_imgs = torch.stack([self.preprocess(img.permute(2, 0, 1)) for img in imgs]) 
        image_features = self.clip_model.encode_image(diff_imgs)
        diff_im_emb_arr = self.normalize(image_features)
        score = self.aesthetic_model(diff_im_emb_arr.float())

        return score


class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def load_aesthetic_model_weights(cache="."):
    weights_fname = "sac+logos+ava1-l14-linearMSE.pth"
    loadpath = os.path.join(cache, weights_fname)

    if not os.path.exists(loadpath):
        url = (
            "https://github.com/christophschuhmann/"
            f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
        )
        r = requests.get(url)

        with open(loadpath, "wb") as f:
            f.write(r.content)

    weights = torch.load(loadpath, map_location=torch.device("cpu"))
    return weights
