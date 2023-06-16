import torch.nn as nn 
import torch 
from torch.nn import CrossEntropyLoss 
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast 
from transformers import AutoModelForCausalLM 
from imagebind.imagebind_model import ModalityType 

from typing import Any, Optional, Tuple, Union
from .qformer import BertConfig, BertLMHeadModel



def get_media_indices(text_list): 
    if isinstance(text_list, torch.Tensor):
        my_list = text_list.cpu().tolist()
    result = []
    for i in range(len(my_list)):
        if i == 0 and my_list[i] < 0:
            result.append(i)
        elif my_list[i] != my_list[i - 1] and my_list[i] < 0:
            result.append(i)
    return result


class AioForConditionalGeneration(PreTrainedModel): 
    def __init__(self, config): 
        super(AioForConditionalGeneration, self).__init__(config)
        if config.vision_model_type == 'imagebind':
            from imagebind import imagebind_model
            self.vision_model = imagebind_model.imagebind_huge(pretrained=True) 
        else: 
            self.vision_model = None 
        self.vision_model.eval() 

        self.language_model = AutoModelForCausalLM.from_pretrained(config.language_model_path)
        self.Qformer, self.query_tokens = self.init_Qformer(
            config.num_query_token, config.vision_hidden_size, config,
        )
        if not config.qformer_text_input: 
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None 
        else:
            self.Qformer.resize_token_embeddings(config.text_vocab)
        self.Qformer.cls = None 

        self.language_input_proj = nn.Linear(config.vision_hidden_size, config.text_hidden_size)
        self.language_output_proj = nn.Linear(config.q_former_hidden_size + config.text_hidden_size, config.text_hidden_size)
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
    def init_Qformer(cls, num_query_token, vision_width, config, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(config.q_former_path)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.is_decoder = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            config.q_former_path, config=encoder_config
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
        no_padding_mask = None,
        non_media_mask = None,
        prompt_mask = None,
        return_dict: Optional[bool] = None, 
        labels = None, 
    ): 
        # get text embedding
        text_tokens_ = input_ids.clone()
        batch_size = input_ids.shape[0] 

        media_token_indices = [
            # [:-1] since we would not use the last token for embedding
            get_media_indices(text_tokens_[i][:-1])
            for i in range(batch_size)
        ]

        text_tokens_[text_tokens_ < 0] = 1  # Not used
        # text_tokens = text_tokens_[:, :-1].contiguous()
        text_embeds = self.get_input_embeddings()(text_tokens_)  # (bsz, seq_len, text_hidden)

        if hasattr(self.language_model, 'transformer') and hasattr(self.language_model.transformer, 'word_embeddings_layernorm'):
            text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

        with torch.no_grad(): 
            # (bsz, image_hidden)
            image_embeds = self.vision_model({ModalityType.VISION: vision_inputs})[ModalityType.VISION] 
        
        image_embeds = image_embeds.unsqueeze(1) # (bsz, image_seq_len, image_hidden)
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        llm_input_image_embeds = self.language_input_proj(image_embeds) 

        image_seq_length = 1

        text_chunk_embeds = []
        for b in range(batch_size):
            start = 0
            result = []
            if len(media_token_indices[b]) > 0:
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(text_embeds[b, start:pos])
                    result.append(llm_input_image_embeds[b, ])
                    start = pos + image_seq_length
            if start < text_embeds.shape[1]:
                result.append(text_embeds[b, start:]) 
            text_chunk_embeds.append(torch.cat(result, dim=0))

        # Actual Input Embeddings (bsz, new_seq_len, text_hidden)
        input_embeds = torch.stack(text_chunk_embeds, dim=0)
        
        outputs = self.language_model.model(
            inputs_embeds=input_embeds, 
            attention_mask=no_padding_mask,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0] # (bsz, seq_len, text_hidden)

        if self.config.qformer_text_input: 
            """
            use question for textual input 
            """
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(vision_inputs.device)
            Qformer_atts = torch.cat([query_atts, prompt_mask],dim=1) 
            # only prompt information is used to weight image information
            context_input_ids = input_ids * prompt_mask
            query_output = self.Qformer.bert(
                context_input_ids[:, 1:],
                attention_mask=Qformer_atts[:, 1:],
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
        else: 
            """
            for image captioning evaluation
            """
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            ) 

        query_output = query_output.last_hidden_state[:,:query_tokens.size(1),:] #(bsz, num_query, q_hidden)
        query_output = torch.mean(query_output, dim=1, keepdim=True) 
        query_output = query_output.repeat(1, hidden_states.size(1), 1) 

        # recall image information 
        output = self.language_output_proj(torch.cat([hidden_states, query_output], dim=2))
        logits = self.language_model.lm_head(output) 

        loss = None 
        if non_media_mask is not None: 
            no_padding_mask = no_padding_mask * (1 - non_media_mask) 
        if prompt_mask is not None: 
            no_padding_mask = no_padding_mask * (1 - prompt_mask) 
        labels[no_padding_mask != 1] = -100

        if labels is not None: 
            shift_logits = logits[..., 1:-1, :].contiguous()
            shift_labels = labels[..., 2:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.text_vocab)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        pixel_values=None, 
        past_key_values=None, 
        attention_mask=None, 
        **model_kwargs,
    ):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "is_decoder": True,
        }


    @torch.no_grad()
    def generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        isdecoder=True,
        **generate_kwargs,
    ):
        """
        Overrides `generate` function to be able to use the model as a conditional generator.
        """
        if attention_mask is None: 
            attention_mask = input_ids.new_ones(*input_ids.shape) 
        
        batch_size = input_ids.size(0)
        media_token_indices = [get_media_indices(input_ids[i]) for i in range(batch_size)] 
        input_ids = input_ids.clone()  # prevent inplace modify
        input_ids[input_ids < 0] = 0  # Not used

        # get text embedding
        text_embeds = self.get_input_embeddings()(input_ids)
        if hasattr(self.language_model, 'transformer') and hasattr(self.language_model.transformer, 'word_embeddings_layernorm'):
            text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds) 
        
        # get visual embedding
        if pixel_values is not None: 
            pixel_values = pixel_values.to(input_ids.device) 

        with torch.no_grad(): 
            # (bsz, image_hidden)
            image_embeds = self.vision_model({ModalityType.VISION: pixel_values})[ModalityType.VISION] 
        
        image_embeds = image_embeds.unsqueeze(1) # (bsz, image_seq_len, image_hidden)

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        llm_input_image_embeds = self.language_input_proj(image_embeds) 

        image_seq_length = 1

        text_chunk_embeds = []
        for b in range(batch_size):
            start = 0
            result = []
            if len(media_token_indices[b]) > 0:
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(text_embeds[b, start:pos])
                    result.append(llm_input_image_embeds[b, ])
                    start = pos + image_seq_length
            if start < text_embeds.shape[1]:
                result.append(text_embeds[b, start:]) 
            text_chunk_embeds.append(torch.cat(result, dim=0))

        # Actual Input Embeddings (bsz, new_seq_len, text_hidden)
        input_embeds = torch.stack(text_chunk_embeds, dim=0) 
        outputs = self.language_model.model(
            inputs_embeds=input_embeds, 
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs[0] # (bsz, seq_len, text_hidden)

        if self.config.qformer_text_input: 
            """
            use question for textual input 
            """
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(pixel_values.device)
            Qformer_atts = torch.cat([query_atts, attention_mask],dim=1) 
            # only prompt information is used to weight image information
            context_input_ids = input_ids * attention_mask 
            query_output = self.Qformer.bert(
                context_input_ids[:, 1:],
                attention_mask=Qformer_atts[:, 1:],
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
        else: 
            """
            for image captioning evaluation
            """
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            ) 

        query_output = query_output.last_hidden_state[:,:query_tokens.size(1),:] #(bsz, num_query, q_hidden)
        query_output = torch.mean(query_output, dim=1, keepdim=True) 
        query_output = query_output.repeat(1, hidden_states.size(1), 1) 

        # recall image information 
        output = self.language_output_proj(torch.cat([hidden_states, query_output], dim=2))
        logits = self.language_model.lm_head(output) 

        # To be updated according to: 
        # https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/generation/utils.py#L1149
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1) 
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1) 

        return input_ids 





