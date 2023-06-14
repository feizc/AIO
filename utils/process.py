import torch 
import numpy as np 

def batchify(batch):
    # collate_fn
    # image = torch.cat([data["image"] for data in batch], dim=0)
    image = [data["vision_inputs"] if data["vision_inputs"] is not None else None for data in batch]
    if all([img is None for img in image]):
        image = None
    else:
        image = torch.cat([img for img in image if img is not None], dim=0)
    
    text = torch.stack([torch.LongTensor(data['input_ids']) for data in batch], dim=0)
    no_padding_mask = torch.stack([torch.LongTensor(data['no_padding_mask']) for data in batch], dim=0)
    non_media_mask = torch.stack([torch.LongTensor(data['non_media_mask']) for data in batch], dim=0)
    prompt_mask = torch.stack([torch.LongTensor(data['prompt_mask']) for data in batch], dim=0)
    
    output_batch = {
        "vision_inputs": image,
        "input_ids": text.long(),
        "labels": text.long().clone(),
        "no_padding_mask": no_padding_mask.long(),
        "non_media_mask": non_media_mask.long(),
        "prompt_mask": prompt_mask.long()        
    }
    return output_batch
    
