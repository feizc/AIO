import torch 
import random 
from PIL import Image, ImageFile 
from torch.utils.data import Dataset
import requests 
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
from urllib.parse import urlparse
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


class LAION400MDataset(Dataset): 
    def __init__(self, file_path, tokenizer, config, prompt_path='./data/prompts.txt'): 
        self.file_path = file_path 
        self.file = open(file_path, 'r', encoding='utf-8') 
        with open(prompt_path, 'r') as f: 
            self.prompts_list = f.readlines() 
        self.tokenizer = tokenizer 
        self.max_text_length = config.max_text_length

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    
    def __len__(self): 
        return 100 
    
    def pad_tokens(self, text): 
        prompt = random.choice(self.prompts_list) 
        prompt_tokens = self.tokenizer.encode('###Human: ' + prompt + ' ###Assistant:  ') 
        text_tokens = self.tokenizer.encode(text) 
        tokens = torch.tensor([-1] + prompt_tokens + text_tokens, dtype=torch.int64) # add image placeholder 
        prompt_length = len(prompt_tokens) 

        padding = self.max_text_length - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1)) 
        else:
            tokens = tokens[:self.max_text_length] 
        
        no_padding_mask = tokens.ge(0)  # mask is zero where we out of sequence 
        tokens[~no_padding_mask] = 0
        tokens[0] = -1
        no_padding_mask[0] = 1
        no_padding_mask = no_padding_mask.long()
        
        non_media_mask = tokens.le(-1) 
        non_media_mask = non_media_mask.long()

        prompt_mask = torch.cat((torch.ones(1 + prompt_length), torch.zeros(self.max_text_length - 1 - prompt_length)), dim=0) 
        prompt_mask = prompt_mask.long()

        return tokens, prompt_length, no_padding_mask, non_media_mask, prompt_mask
        
    
    def __getitem__(self, index): 
        data = self.file.readline() 
        if not data: 
            self.file = open(self.file_path, 'r') 
            data = self.file.readline() 
        
        url, text = data.strip().split('\t') 
        
        # image
        if 'http' in url:
            domain = urlparse(url).netloc
            url = url.replace(domain, 'p.vip.sankuai.com')+'@384w'
            session = requests.Session()
            retry = Retry(connect=3, read=3, redirect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            response = session.get(url)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(str(url)) 

        img = img.convert("RGB") 
        img = self.image_transform(img) 

        # text 
        tokens, prompt_length, no_padding_mask, non_media_mask, prompt_mask = self.pad_tokens(text) 
        prompt_length = torch.Tensor([prompt_length]).long()

        return {
            "vision_inputs": img, 
            "input_ids": tokens,
            # "prompt_length": prompt_length,
            "no_padding_mask": no_padding_mask,
            "non_media_mask": non_media_mask,
            "prompt_mask": prompt_mask,
        }


def build_train_valid_datasets(input_file, tokenizer, config, data_type='LAION'): 
    assert len(input_file) == 2 
    if data_type == 'LAION': 
        train_dataset = LAION400MDataset(file_path=input_file[0], tokenizer=tokenizer, config=config) 
        valid_dataset = LAION400MDataset(file_path=input_file[1], tokenizer=tokenizer, config=config) 
    return (train_dataset, valid_dataset) 

