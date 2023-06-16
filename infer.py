import torch 
from transformers import AutoTokenizer 

import utils.data as data
from model import AioForConditionalGeneration 
from PIL import Image 
from utils import image_transforms_build 

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_and_tokenizer(pretrained_ckpt):
    model = AioForConditionalGeneration.from_pretrained(pretrained_ckpt,)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt,) 
    return model, tokenizer 


def generate(input_texts, image_lists, model, tokenizer, **generate_kwargs): 
    img = Image.open(image_lists).convert('RGB') 
    img = image_transforms_build(224)(img) 

    input_ids = tokenizer.encode(input_texts) 
    return model.generate(
        pixel_values=img,
        input_ids=input_ids,
    )


if __name__ == "__main__": 
    input_texts = "The following is a conversation between a curious human and AI assistant."
    image_lists = ['dog.jpg'] 

    base_model_path = 'ckpt' 
    model, tokenizer = get_model_and_tokenizer(base_model_path) 
    sentence = generate(
        input_texts, image_lists, model, tokenizer,
        max_length=512, top_k=5, do_sample=True,
    )
    print(sentence)
