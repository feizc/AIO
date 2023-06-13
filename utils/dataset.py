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
    def __init__(self, file_path, tokenizer, prompt_path='./data/prompts.txt'): 
        self.file_path = file_path 
        self.file = open(file_path, 'r', encoding='utf-8') 
        with open(prompt_path, 'r') as f: 
            self.prompts_list = f.readlines() 
        self.tokenizer = tokenizer 

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
    
    def __getitem__(self, index): 
        data = self.file.readline() 
        if not data: 
            self.file = open(self.file_path, 'r') 
            data = self.file.readline() 
        
        url, text = data.strip().split('\t') 
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
        return {
            "input_ids"
            "prompt_length"
            "no_padding_mask"
            "non_media_mask"
            "prompt_mask"
        }