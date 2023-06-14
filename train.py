import torch 
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, AutoTokenizer 
from argparse import ArgumentParser

from model import AioConfig, AioForConditionalGeneration 
from utils import batchify, build_train_valid_datasets 



class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        sampler = DistributedSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=batchify)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        dataset = self.eval_dataset
        sampler = DistributedSampler(dataset, shuffle=False)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=batchify)



def main(): 
    parser = ArgumentParser() 
    parser.add_argument("--train_data_path", type=str, default="data/", help="Path of the train dataset")
    parser.add_argument("--valid_data_path", type=str, default="data/", help="Path of the validation dataset")
    parser.add_argument("--language_model_path", type=str, default="ckpt/llama", help="Path of the language model")
    parser.add_argument('--train-epochs', type=int, default=3,
                    help='Total number of epochs to train over all '
                    'training runs.') 
    parser.add_argument('--use-lora', type=bool, default=False, help='LORA.')
    args = parser.parse_args() 

    config = AioConfig(language_model_path=args.language_model_path) 
    model = AioForConditionalGeneration(config) 

    tokenizer = AutoTokenizer.from_pretrained(args.language_model_path, use_fast=False) 
    tokenizer.pad_token='[PAD]' 

    if args.use_lora: 
        pass 
    else:
        for name, param in model.named_parameters():
            if 'language_model' in name: 
                param.requires_grad = False 
            elif 'vision_model' in name: 
                param.requires_grad = False 
            else: 
                param.requires_grad = True 
    
    model.train() 
    train_data, valid_data = build_train_valid_datasets(
        input_file=[args.train_data_path, args.valid_data_path],
        tokenizer=tokenizer, config=config,
    ) 
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )








if __name__ == '__main__':
    main()