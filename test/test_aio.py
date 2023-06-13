import torch 
from model import AioConfig, AioForConditionalGeneration 
from imagebind.imagebind_model import ModalityType
from utils import data

device = "cuda:1" if torch.cuda.is_available() else "cpu"
config = AioConfig()
model = AioForConditionalGeneration(config)
model = model.to(device) 

image_paths = ['dog.jpg'] 
inputs = data.load_and_transform_vision_data(image_paths, device)  
text_ids = torch.Tensor([[-1, 1, 2, 5]]).long().to(device) 
attention_mask = torch.Tensor([[1, 1, 1, 0]]).long().to(device) 
output = model(vision_inputs=inputs, input_ids=text_ids, attention_mask=attention_mask)
print(output.loss)
