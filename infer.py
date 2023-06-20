import torch 
from transformers import AutoTokenizer 
import torch.nn.functional as F 

import utils.data as data
from model import AioForConditionalGeneration 
from PIL import Image 
from utils import image_transforms_build 
from llama import LlamaTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits



def get_model_and_tokenizer(pretrained_ckpt):
    model = AioForConditionalGeneration.from_pretrained(pretrained_ckpt,)
    #tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt,) 
    tokenizer = LlamaTokenizer.from_pretrained('./ckpt/llama', use_fast=False) 
    tokenizer.pad_token='[PAD]'
    return model, tokenizer 


def generate(input_texts, image_lists, model, tokenizer, max_length, min_length, temperature, top_k, no_sample, top_p, batch_size): 
    img = Image.open(image_lists[0]).convert('RGB') 
    img = image_transforms_build(224)(img) 
    img = img.unsqueeze(0) 
    
    input_ids = tokenizer.encode(input_texts) 
    input_ids = torch.LongTensor(input_ids).unsqueeze(0)
    eos_ids = tokenizer.eos_token_id
    for i in range(max_length): 
        logits = model.generate(
            pixel_values=img,
            input_ids=input_ids,
        )[:, -1, :]
        # logits = logits.squeeze() 
        logits = logits / temperature 
        if i < min_length:
            logits[:, eos_ids] = -1e9 
        lm_logits = torch.cat([top_filtering(logits[j], top_k=top_k, top_p=top_p).unsqueeze(0) for j in range(batch_size)], dim=0)
        
        probs = F.softmax(lm_logits, dim=-1)
        prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
        input_ids = torch.cat([input_ids, prev.cpu()], dim=-1) 
    
    decode_result = []
    for i in range(0, batch_size):
        temp = input_ids[i, :].cpu().tolist()
        temp1 = []
        for j in temp:
            if j == eos_ids:
                break
            temp1.append(j)
        decode_result.append((tokenizer.decode(temp1, skip_special_tokens=True) + "\n").replace("1.0 ", "").replace("0.0 ", ""))
    return decode_result



if __name__ == "__main__": 
    input_texts = "The following is a conversation between a curious human and AI assistant."
    image_lists = ['dog.jpg']

    base_model_path = 'output' 
    model, tokenizer = get_model_and_tokenizer(base_model_path) 
    model = model.eval() 

    with torch.no_grad():
        sentence = generate(
            input_texts, image_lists, model, tokenizer,
            max_length=10, min_length=2, temperature=0.9, top_k=5, no_sample=False, 
            top_p=0, batch_size=1, 
        )
    print(sentence)
