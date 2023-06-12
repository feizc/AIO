import torch.nn as nn 
import torch 
from transformers.modeling_utils import PreTrainedModel
from transformers import BertConfig, BertLMHeadModel 
from transformers import AutoModelForCausalLM 
from imagebind.imagebind_model import ModalityType 
import utils.data as data 


class AioForConditionalGeneration(PreTrainedModel): 
    def __init__(self, config): 
        if config.vision_model_type == 'imagebind':
            from imagebind import imagebind_model
            self.vision_model = imagebind_model.imagebind_huge(pretrained=True) 
        else:
            self.vision_model = None 
        self.vision_model.eval() 

        self.language_model = AutoModelForCausalLM.from_pretrained(config.language_model_path)
        self.Qformer, self.query_tokens = self.init_Qformer(
            config.num_query_token, config.vision_hidden_size
        )
        self.Qformer.cls = None 

        self.language_input_proj = nn.Linear(config.vision_hidden_size, config.text_hidden_size)
        self.language_output_proj = nn.Linear(config.q_former_hidden_size, config.text_hidden_size)
        self.config = config 
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_decoder(self):
        return self.language_model.get_decoder()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("ckpt/bert")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "ckpt/bert", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens 
    
    def forward(
        self,
        vision_inputs,
        input_ids: torch.FloatTensor,
    ): 
        # get text embedding
        text_tokens_ = input_ids.clone()
        batch_size = input_ids.shape[0] 

        text_tokens_[text_tokens_ < 0] = 1  # Not used
        # text_tokens = text_tokens_[:, :-1].contiguous()
        text_embeds = self.get_input_embeddings()(text_tokens_) 

        if hasattr(self.language_model, 'transformer') and hasattr(self.language_model.transformer, 'word_embeddings_layernorm'):
            text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

        with torch.no_grad():
            image_embeds = self.vision_model({ModalityType.VISION: vision_inputs})[ModalityType.VISION] 
        
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_features = self.abstractor(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )["last_hidden_state"]
        
        







