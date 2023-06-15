import torch 
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, AutoTokenizer 
from argparse import ArgumentParser
from transformers.training_args import TrainingArguments

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
    
    # Data
    parser.add_argument("--train_data_path", type=str, default="data/", help="Path of the train dataset")
    parser.add_argument("--valid_data_path", type=str, default="data/", help="Path of the validation dataset")
    parser.add_argument("--language_model_path", type=str, default="ckpt/llama", help="Path of the language model")
    
    # Training
    parser.add_argument('--lr', type=float, default=2e-5,
                    help='Initial learning rate. Depending on decay style '
                    'and initial warmup, the learing rate at each '
                    'iteration would be different.') 
    parser.add_argument('--num-warmup-steps', type=int, default=50,
                    help='The number of warmup steps.')
    parser.add_argument('--do-train', action='store_true', default=True,
                    help='Whether to do training.')  
    parser.add_argument('--save-interval', type=int, default=None,
                    help='Number of iterations between checkpoint saves.')
    parser.add_argument('--eval-iters', type=int, default=100,
                    help='Number of iterations to run for evaluation'
                    'validation/test for.')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                    help='Weight decay coefficient for L2 regularization.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                        help='The gradient accumulation steps.')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient clipping based on global L2 norm.')
    parser.add_argument('--bf16', action='store_true', default=True,
                    help='Run model in bfloat16 mode.')
    parser.add_argument('--ddp-find-unused-parameters', action='store_true',
                    help='unused parameters finding.')

    parser.add_argument('--train-epochs', type=int, default=3,
                    help='Total number of epochs to train over all '
                    'training runs.') 
    parser.add_argument('--micro-batch-size', type=int, default=4,
                    help='Batch size per model instance (local batch size). '
                    'Global batch size is local batch size times data '
                    'parallel size times number of micro batches.')
    parser.add_argument('--use-lora', type=bool, default=False, help='LORA.')
    parser.add_argument('--save-path', type=str, default='output',
                    help='Output directory to save checkpoints to.')
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank')
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
        args=TrainingArguments(
            learning_rate=args.lr,
            warmup_steps=args.num_warmup_steps,
            do_train=args.do_train,
            num_train_epochs=args.train_epochs,
            output_dir=args.save_path,
            save_strategy='steps',
            save_steps=args.save_interval,
            evaluation_strategy='steps',
            eval_steps=args.eval_iters,
            per_device_train_batch_size=args.micro_batch_size,
            max_grad_norm=args.clip_grad,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            fp16=not args.bf16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.eval_iters//4,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
            report_to=['tensorboard'],
        )
    )

    trainer.train() 
    model.save_pretrained(args.save_path)


if __name__ == '__main__':
    main()