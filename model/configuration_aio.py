from transformers.configuration_utils import PretrainedConfig 


class AioConfig(PretrainedConfig): 
    # model_type = "aio_config" 
    def __init__(
        self, 
        vision_model_type='imagebind', # clip, vit
        num_query_token=10,
        q_former_hidden_size=768,
        qformer_text_input=True,
        vision_hidden_size=1024,
        text_hidden_size=5120, 
        text_vocab=32000,
        max_text_length=64,
        language_model_path='./ckpt/llama', 
        q_former_path='./ckpt/bert',
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.vision_model_type = vision_model_type 
        self.q_former_hidden_size = q_former_hidden_size 
        self.qformer_text_input = qformer_text_input
        self.vision_hidden_size = vision_hidden_size 
        self.text_hidden_size = text_hidden_size
        self.language_model_path = language_model_path 
        self.num_query_token = num_query_token
        self.text_vocab = text_vocab
        self.max_text_length=max_text_length 
        self.q_former_path = q_former_path

    

