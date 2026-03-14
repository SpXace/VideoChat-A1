
import copy
import numpy as np
from transformers import AutoModel, AutoProcessor
from transformers import AutoTokenizer, AutoModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import torch

from decord import cpu, VideoReader 
from torch import distributed as dist
from tqdm import tqdm
import json
from tqdm import tqdm
from all_model_util import *
import torch
import os
import json
import random
import argparse
import time
import pandas as pd
import re
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

def get_anno(anno_path):
    anno = json.load(open(anno_path, 'r'))
    return anno[0]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Qwen2_5_7bAgent:
    def __init__(self, device="cuda:0"): 
        model_path = 'yourpath/ckpt/Qwen2.5-VL-7B-Instruct'
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.device = device        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=self.device)   
    
    def get_model_name(self):
        return 'qwen_7b'

    def get_text_answer(self, text):
        messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        images, videos = None, None
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def get_answer(self, video_path, text, sample_idx=None, multi_image_path= None):

        if multi_image_path:
            messages = [
                {"role": "user", 
                "content": [{"type": "image", "image": video_path} for video_path in multi_image_path]}]
            messages[0]['content'].append({"type": "text", "text": text})
            images, videos = process_vision_info(messages)
        
    
        elif video_path == None:
            messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
            images, videos = None, None
            
        else:
            messages = [[{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": text}]}]]
            images, videos = process_vision_info(messages, sample_idx)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text