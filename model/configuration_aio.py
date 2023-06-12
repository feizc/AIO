from transformers.configuration_utils import PretrainedConfig 


class AioConfig(PretrainedConfig): 
    model_type = "aio_config" 
    def __init__(
        self, 
        vision_model_type='imagebind',
        num_query_token=10,
        q_former_hidden_size=768,
        vision_hidden_size=1024,
        text_hidden_size=5120,
        language_model_path='./ckpt/llama',
    ): 
        self.vision_model_type = vision_model_type 
        self.q_former_hidden_size = q_former_hidden_size
        self.vision_hidden_size = vision_hidden_size 
        self.text_hidden_size = text_hidden_size
        self.language_model_path = language_model_path 
        self.num_query_token = num_query_token

    

