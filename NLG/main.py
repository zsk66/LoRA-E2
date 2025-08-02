import os
import warnings
import re
import numpy as np
import pandas as pd
import typing as tp
from tqdm import tqdm
import torch
from datasets import load_dataset, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, TrainingArguments,Seq2SeqTrainingArguments)
from sklearn.metrics import accuracy_score
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer,SFTConfig
# import wandb
import logging
from transformers import Trainer
from NLG.utils import new_update_layer,setup_seed
from transformers import TrainerCallback
from peft.tuners.lora.layer import LoraLayer
import time
os.environ["WANDB_DISABLED"] = "true"



class LossLoggingCallback(TrainerCallback):
    """Callback for logging training loss to a CSV file during model training.

    This callback tracks the training loss at specified intervals and writes
    the results to a CSV file for later analysis and visualization.
    """
    def __init__(self, log_file="training_loss.csv"):
        self.log_file = log_file
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append({"step": state.global_step, "loss": logs["loss"]})
            if state.global_step % 10 == 0:
                df = pd.DataFrame(self.losses)
                df.to_csv(self.log_file, index=False)

class LoRAGaussSeidelTrainer(Trainer):
    """Custom Trainer implementing Gauss-Seidel optimization for LoRA parameters.

    This Trainer supports multiple LoRA training modes with different parameter
    update strategies, allowing for flexible optimization of low-rank adaptation layers.
    """
    def __init__(
        self,
        LoRA_mode,
        **kwargs,
    ):
        """Initialize the LoRAGaussSeidelTrainer with specified LoRA mode.

        Args:
            LoRA_mode (str): LoRA training mode to use. Valid options include:
                - 'LoRA-A': Alternating optimization of A and B matrices
                - 'LoRA-S': Simultaneous optimization of all parameters
                - 'LoRA-F': Only optimize B matrix
            **kwargs: Additional keyword arguments passed to the base Trainer
        """
        super().__init__(**kwargs)
        self.LoRA_mode=LoRA_mode
    def training_step(self, model, inputs, num_items_in_batch):
        """Perform a single training step with LoRA-specific parameter updates

        Args:
            model: The model being trained
            inputs: Dictionary of input data
            num_items_in_batch: Number of items in the batch

        Returns:
            torch.Tensor: Training loss value
        """
        if self.LoRA_mode == 'LoRA-A':
            original_lr = self.args.learning_rate

            # Train B matrix (freeze A matrix)
            for name, param in model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = False
                if 'lora_B' in name:
                    param.requires_grad = True

            loss_B = super().training_step(model, inputs, num_items_in_batch)
            
            # Reset learning rate to original value
            for group in self.optimizer.param_groups:
                group['lr'] = original_lr
            
            # Train A matrix (freeze B matrix)
            for name, param in model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = True
                if 'lora_B' in name:
                    param.requires_grad = False

            # Second training step to update A
            loss_A = super().training_step(model, inputs, num_items_in_batch)
    
            return loss_A
        elif self.LoRA_mode == 'LoRA-S':
            # Simultaneous training of all parameters
            loss = super().training_step(model, inputs, num_items_in_batch)
            return loss

        elif self.LoRA_mode == 'LoRA-F':
            # Single training step to update B
            for name, param in model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = False
                if 'lora_B' in name:
                    param.requires_grad = True

            # Single training step to update B
            loss_B = super().training_step(model, inputs, num_items_in_batch)
            return loss_B

        else:
            raise ValueError("Invalid LoRA mode")
        


alpha = 16
seed = 0
ranks = [8]
seeds = [0]
for seed in seeds:
    setup_seed(seed)
    for rank in ranks:
        print(f'seed={seed}, rank={rank}')
        # Set up Weights & Biases (wandb) for logging
        # os.environ["WANDB_PROJECT"] = "llama_gsm8k"
        # os.environ["WANDB_LOG_MODEL"] = "end"  # 或者 "checkpoint"，如果你想在每次保存模型时上传
        # os.environ["WANDB_WATCH"] = "false"
        # wandb.login(key="7c23dbfd00e280075d52da5dc0587bbc73d4805d")

        # Set up logging
        warnings.filterwarnings("ignore")
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")


        model_name = "NLG/models/Llama-2-7B-hf"

        # model & tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
            model.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # tokenizer.save_pretrained('/data02/zsk/models/ft_res_models/2025-05-17-00-51-01/')

        address = 'NLG/results'
        LoRA_mode = 'LoRA-A'
        dataset_name = "GSM8K"
        init_lora = 'stable_init'    
        use_rslora = False
        use_dora = False
        # load meta-math-qa data 110000 samples, 100k for train, 10k for eval 
        def load_meta_math(max_tokens=512):
            dataset = load_dataset("/data02/zsk/data/MetaMathQA", split="train")
            from transformers import AutoTokenizer
            tokenizer_d = AutoTokenizer.from_pretrained(model_name)

            def preprocess(data):
                return {
                    "x": f'Q: {data["query"]}\nA: ',
                    "y": data["response"].split("\nThe answer is:")[0],
                }

            train_samples = []
            eval_samples = []
            count = 0
            dataset.shuffle(seed=0)
            from tqdm import tqdm
            bar = tqdm(dataset, total=110000)
            total = 0
            ok = 0
            for sample in dataset:
                total += 1
                temp = preprocess(sample)
                if (
                    len(tokenizer_d(temp["x"] + " " + temp["y"])["input_ids"]) >= max_tokens
                    or "GSM" not in sample["type"]
                ):
                    continue
                bar.update(1)
                bar.set_description(f"ok: {ok}/{total}")
                ok += 1
                processed_sample = preprocess(sample)
                if count < 100000:  # First 100,000 samples for training
                    train_samples.append(processed_sample)
                elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
                    eval_samples.append(processed_sample)
                elif count >= 110000:  # Stop processing after collecting enough samples
                    break
                count += 1
            # convert to hf dataset
            train_set = Dataset.from_list(train_samples)
            eval_set = Dataset.from_list(eval_samples)
            return train_set, eval_set

        train_set, eval_set = load_meta_math()

        def preprocess_dataset(
            dataset: tp.Union[Dataset, tp.List[tp.Tuple[str, str]], tp.List[tp.Dict[str, str]]]
        ) -> Dataset:
            if isinstance(dataset, list) and isinstance(dataset[0], tuple):
                dataset = Dataset.from_pandas(pd.DataFrame(dataset, columns=["x", "y"]))
            elif isinstance(dataset, list) and isinstance(dataset[0], dict):
                dataset = Dataset.from_dict(
                    {k: [dic[k] for dic in dataset] for k in dataset[0]}
                )
            elif isinstance(dataset, dict):
                dataset = Dataset.from_dict(dataset)
            elif isinstance(dataset, Dataset):
                pass
            else:
                raise ValueError("Wrong format")
            return dataset

        train_dataset = preprocess_dataset(train_set)
        valid_dataset = preprocess_dataset(eval_set)

        def causalLMEncode(example, tokenizer, max_length=-1, ignore_masked_token=True):
            is_list_input = isinstance(example["x"], list)
            # Combine text and add EOS token
            combined_text = (
                [
                    x + " " + y + tokenizer.eos_token
                    for (x, y) in zip(example["x"], example["y"])
                ]
                if is_list_input
                else example["x"] + " " + example["y"] + tokenizer.eos_token
            )
            # Tokenize combined text
            encodings = tokenizer(
                combined_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length if max_length != -1 else None,
            )
            # Calculate input text length in tokens
            input_text_length = (
                [
                    len(tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
                    for i in range(len(example["x"]))
                ]
                if is_list_input
                else len(tokenizer(example["x"], return_tensors="pt")["input_ids"][0])
            )
            if input_text_length[0] >= max_length:
                print(
                    f"Input text length >= max_length: {input_text_length} >= {max_length}. "
                    "Consider increasing max_length to avoid truncation."
                )
            # Create labels
            labels = encodings["input_ids"].clone()
            if is_list_input:
                for i, l in enumerate(input_text_length):
                    labels[i, :l] = -100
            else:
                labels[0, :input_text_length] = -100
            if ignore_masked_token:
                labels[encodings["attention_mask"] == 0] = -100
            # Update example dictionary
            results = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
                # "input_text_length": input_text_length,
            }

            return results

        def transform_dataset(tokenizer, dataset, max_length, model_type='CausalLM'):
            if model_type == "CausalLM":
                dataset.set_transform(lambda x: causalLMEncode(x, tokenizer, max_length))
            else:
                raise ValueError("Wrong model type")
            return dataset

        train_dataset, valid_dataset = transform_dataset(tokenizer, train_dataset, max_length=1024), transform_dataset(tokenizer, valid_dataset, max_length=1024)





        LoraLayer.update_layer = new_update_layer
        # Define PEFT LoRA Configuration
        peft_config = LoraConfig(
            lora_alpha=alpha,
            lora_dropout=0.1,
            r=rank,
            init_lora_weights=init_lora,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora = use_rslora,
            use_dora = use_dora,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        # Training Arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir="./logs",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            optim="adamw_torch",
            logging_steps=1,
            save_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            bf16=True,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            eval_strategy="no",
            save_total_limit=2,
            # report_to="wandb",
            remove_unused_columns=False
        )


        loss_address = address + "loss/" + dataset_name + "_" + LoRA_mode + '_' + str(init_lora) + '_rsLoRA='+str(use_rslora) + '_DoRA=' + str(use_dora) + "_rank=" + str(rank) + "_alpha="+ str(alpha) + "_seed=" + str(seed)  + "_training_loss.csv"
        loss_logging_callback = LossLoggingCallback(log_file=loss_address)
        # Initialize the Trainer and start fine-tuning
        trainer = LoRAGaussSeidelTrainer(
            LoRA_mode=LoRA_mode,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            args=training_args,
            callbacks=[loss_logging_callback],  
        )
        trainer.train()

        # Save the model locally before pushing
        savepath="NLG/models/fine_tuned_models/"+ LoRA_mode + '_' + str(init_lora) + '_rsLoRA='+str(use_rslora) + '_DoRA=' + str(use_dora) + '_rank=' + str(rank) + '_alpha=' + str(alpha) + '_seed='+str(seed) 
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        trainer.save_model(savepath)
        tokenizer.save_pretrained(savepath)

        # Push the fine-tuned model to Hugging Face Hub
        # model.push_to_hub(
        #     "#Your repo",
        #     # use_auth_token=hf_token,
        #     commit_message="fine-tuned on GSM-8k in A800",
        #     private=False
        # )