import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from NLU.utils import stable_init, new_update_layer,setup_seed, kaiming_init
import sys
from transformers import Trainer, Seq2SeqTrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from transformers import TrainerCallback
from convert import *
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
import time
import math
from transformers.trainer_utils import speed_metrics,EvalLoopOutput,EvalPrediction,has_length,denumpify_detensorize
from transformers.trainer_pt_utils import EvalLoopContainer,find_batch_size,IterableDatasetShard
import re
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global dataset_name
global tokenizer

labels_maxlen_each_task = {
    'cola': 4,
    'stsb': 4,
    'sst2': 2,
    'mrpc': 2,
    'qqp': 6,
    'mnli': 3,
    'qnli': 2,
    'rte': 2,
    'wnli': 2,
}

class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_file="training_loss.csv"):
        self.log_file = log_file
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append({"step": state.global_step, "loss": logs["loss"]})
            if state.global_step % 10 == 0:
                df = pd.DataFrame(self.losses)
                df.to_csv(self.log_file, index=False)

class AccuracyLoggingCallback(TrainerCallback):
    def __init__(self, log_file="validation_accuracy.csv"):
        self.log_file = log_file
        self.accuracies = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if "eval_accuracy" in metrics:
            self.accuracies.append({"step": state.global_step, "accuracy": metrics["eval_accuracy"]})
            df = pd.DataFrame(self.accuracies)
            df.to_csv(self.log_file, index=False)
        elif "eval_matthew_corrcoef" in metrics:
            self.accuracies.append({"step": state.global_step, "matthew_corrcoef": metrics["eval_matthew_corrcoef"]})
            df = pd.DataFrame(self.accuracies)
            df.to_csv(self.log_file, index=False)
        elif "eval_Pearson_correlation" in metrics:
            self.accuracies.append({"step": state.global_step, "Pearson_correlation": metrics["eval_Pearson_correlation"]})
            df = pd.DataFrame(self.accuracies)
            df.to_csv(self.log_file, index=False)
def compute_metrics(pred):
    logits = pred.predictions[0] 
    pred_ids = logits.argmax(axis=-1) 
    labels = pred.label_ids
    labels = np.where(labels == -100, 0, labels)
    
    accuracy = accuracy_score(labels.flatten(), pred_ids.flatten())
    return {"accuracy": accuracy}






class LoRAGaussSeidelTrainer(Trainer):
    def __init__(self, LoRA_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LoRA_mode = LoRA_mode
    def training_step(self, model, inputs, num_items_in_batch):
        if self.LoRA_mode == 'LoRA-A':
            original_lr = self.args.learning_rate

            for name, param in model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = False
                if 'lora_B' in name:
                    param.requires_grad = True

            loss_B = super().training_step(model, inputs, num_items_in_batch)
            
            for group in self.optimizer.param_groups:
                group['lr'] = original_lr
            
            for name, param in model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = True
                if 'lora_B' in name:
                    param.requires_grad = False

            loss_A = super().training_step(model, inputs, num_items_in_batch)
    
            return loss_A
        elif self.LoRA_mode == 'LoRA-S':
            loss = super().training_step(model, inputs, num_items_in_batch)
            return loss

        elif self.LoRA_mode == 'LoRA-F':
            for name, param in model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = False
                if 'lora_B' in name:
                    param.requires_grad = True
            loss_B = super().training_step(model, inputs, num_items_in_batch)
            return loss_B
        else:
            raise ValueError("Invalid LoRA mode")
        
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or self.is_fsdp_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size
        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        if(dataset_name == 'stsb'):
            pearson_x = []
            pearson_y = []
        elif(dataset_name =='cola'):
            matthew_true = []
            matthew_pred = []
        else:
            all_corrects = []

        metrics = {}
        eval_set_kwargs = {}

        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            # Calculate accuracy for this batch
            if logits is not None and labels is not None:
                if(dataset_name =='stsb'):
                    def map_to_string_labels(tensor_labels, tokenizer):
                        string_labels = []
                        for i in range(tensor_labels.shape[0]): 
                            string_row = [tokenizer.decode([idx.item()]) for idx in tensor_labels[i]]
                            try:
                                string_row = float(string_row[1])
                            except ValueError:
                                string_row = -1.0
                            string_labels.append(string_row)
                        return string_labels
                    predictions = torch.argmax(logits[0], dim=-1)  # Get predicted class
                    decode_x = np.array(map_to_string_labels(labels, tokenizer))
                    decode_y = np.array(map_to_string_labels(predictions, tokenizer))
                    pearson_x.extend(decode_x.tolist())
                    pearson_y.extend(decode_y.tolist())
                elif(dataset_name =='cola'):
                    def map_to_string_labels(tensor_labels, tokenizer):
                        string_labels = []
                        for i in range(tensor_labels.shape[0]): 
                            string_row = [tokenizer.decode([idx.item()]) for idx in tensor_labels[i]]
                            if(string_row[0] == 'acceptable'):
                                string_labels.append(1)
                            elif(string_row[0] == 'unacceptable'):
                                string_labels.append(0)
                            else:
                                string_labels.append(-1)
                        return string_labels
                    predictions = torch.argmax(logits[0], dim=-1)  # Get predicted class
                    decode_true = np.array(map_to_string_labels(labels, tokenizer))
                    decode_pred = np.array(map_to_string_labels(predictions, tokenizer))
                    matthew_true.extend(decode_true[decode_pred != -1].tolist())
                    matthew_pred.extend(decode_pred[decode_pred != -1].tolist())
                elif(dataset_name =='qqp'):
                    def map_to_string_labels(tensor_labels, tokenizer):
                        string_labels = []
                        for i in range(tensor_labels.shape[0]):  
                            string_row = [tokenizer.decode([idx.item()]) for idx in tensor_labels[i]]
                            end_idx = string_row.index("</s>")
                            filtered_string = ''.join(string_row[:end_idx])
                            if(filtered_string == 'duplicate'):
                                string_labels.append(1)
                            elif(filtered_string == 'not_duplicate'):
                                string_labels.append(0)
                            else:
                                string_labels.append(0)
                        return string_labels
                    predictions = torch.argmax(logits[0], dim=-1)  # Get predicted class
                    decode_true = np.array(map_to_string_labels(labels, tokenizer))
                    decode_pred = np.array(map_to_string_labels(predictions, tokenizer))
                    matches = decode_true == decode_pred
                    correct = np.sum(matches)
                    all_corrects.append(correct) 
                else:
                    predictions = torch.argmax(logits[0], dim=-1)  # Get predicted class
                    correct = (torch.all(predictions == labels, dim=1)).sum().item()# Count correct predictions
                    all_corrects.append(correct)  # Store the batch correct predictions

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After gathering function, reset for metrics collection
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        # Gather all remaining tensors
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Final metrics
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None and not self.args.batch_eval_metrics:
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )

        # To make the metrics JSON serializable
        metrics = denumpify_detensorize(metrics)

        if (dataset_name =='stsb'):
            x = np.array(pearson_x)
            y = np.array(pearson_y)
            from scipy.stats import pearsonr
            pearson_correlation, p_value = pearsonr(x, y)
            metrics[f"{metric_key_prefix}_Pearson_correlation"] = pearson_correlation
        elif(dataset_name =='cola'):
            from sklearn.metrics import matthews_corrcoef
            matthew_corrcoef = matthews_corrcoef(matthew_true, matthew_pred)
            metrics[f"{metric_key_prefix}_matthew_corrcoef"] = matthew_corrcoef
        else:
            total_correct = sum(all_corrects)
            metrics[f"{metric_key_prefix}_accuracy"] = total_correct / num_samples

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix keys with metric_key_prefix
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
def convert_single_slice(ds_name, slice):
    ds_cfg=find_ds_config(ds_name)
    label_names = ds_cfg.label_classes
    processed_sample = glue(slice, ds_name, label_names)
    return processed_sample

def convert_dataset(dataset):
    ds_cfg=find_ds_config(dataset_name)
    label_names = ds_cfg.label_classes
    inputs= []
    targets=[]

    def dict_iterator(column_dict):
        columns = list(column_dict.keys()) 
        values = list(column_dict.values()) 
        for row in zip(*values):  
            row_dict = dict(zip(columns, row))
            yield row_dict  
    
    iterator = dict_iterator(dataset)
    for slice in iterator:
        processed_sample = glue(slice, dataset_name, label_names)
        inputs.append(processed_sample['inputs'])
        targets.append(processed_sample['targets'])
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=labels_maxlen_each_task[dataset_name], truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class EvaluateAtEndOfEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        return control



dataset_names = ['sst2','mrpc', 'qqp','stsb', 'qnli', 'rte', 'wnli','mnli', 'cola']

init_A_B = 'stable_init'  

LoRA_modes = ['LoRA-A', 'LoRA-S', 'LoRA-F']

use_dora = False
use_rslora = False
ranks = [8]
alpha = 16
seeds = [1,2,3]
batch_size = 32
eval_batch_size = 32
learning_rate = 1e-4
for dataset_name in dataset_names:
    for rank in ranks:
        if dataset_name == 'cola' or dataset_name == 'mrpc' or dataset_name == 'stsb':
            batch_size = 4
        else:
            batch_size = 32
        for LoRA_mode in LoRA_modes:
            for seed in seeds:
                setup_seed(seed)
                print(f"dataset_name: {dataset_name}")
                print(f"LoRA_mode: {LoRA_mode}")
                print(f"seed: {seed}")
                # for dataset_name in datasets_set:
                if init_A_B == 'stable_init':
                    LoraLayer.init_A_B_in_my_way = stable_init
                elif init_A_B =='kaiming_init':
                    LoraLayer.init_A_B_in_my_way = kaiming_init
                LoraLayer.update_layer = new_update_layer

                dataset = load_dataset("NLU/data/GLUE", dataset_name)
                if "test" in dataset:
                    dataset.pop("test")

                model_name = "NLU/models/t5_base"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                if LoRA_mode == 'LoRA-A':
                    init_str = 'my_init'
                else: 
                    init_str = True
                config = LoraConfig(
                    r=rank,  
                    lora_alpha=alpha,
                    target_modules=["q", "v"],  
                    lora_dropout=0.1,
                    init_lora_weights=init_str,
                    bias="none",
                    task_type="SEQ_2_SEQ_LM",
                    use_dora = use_dora,
                    use_rslora = use_rslora
                )
                model = get_peft_model(model, config)




                tokenized_dataset = dataset.map(
                    convert_dataset,
                    batched=True
                )
                training_args = Seq2SeqTrainingArguments(
                    output_dir="./results",
                    num_train_epochs=1,
                    learning_rate=learning_rate,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=eval_batch_size,
                    save_steps=10_000,
                    save_total_limit=2,
                    eval_strategy="steps",
                    eval_steps=1000,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_dir="./logs",
                    logging_steps=1,
                    eval_do_concat_batches = False, 
                    # fp16=torch.cuda.is_available(),
                )


                if dataset_name == "mnli":
                    eval_dataset = tokenized_dataset["validation_matched"] 
                else:
                    eval_dataset = tokenized_dataset["validation"]

                address = 'NLU/results'
                loss_address = address + "loss/" + dataset_name + "_" + LoRA_mode + '_' + init_A_B + '_DoRA_' + str(use_dora) +  '_rsLoRA_' + str(use_rslora) + "_rank=" + str(rank) + "_alpha="+ str(alpha) + "_seed=" + str(seed)  + "_training_loss.csv"
                accuracy_address = address + "metric/" + dataset_name + "_" + LoRA_mode+ '_' + init_A_B + '_DoRA_' + str(use_dora) +  '_rsLoRA_' + str(use_rslora) + "_rank=" + str(rank)  + "_alpha="+ str(alpha) + "_seed=" + str(seed)  + "_validation_accuracy.csv"
                loss_logging_callback = LossLoggingCallback(log_file=loss_address)
                accuracy_logging_callback = AccuracyLoggingCallback(log_file=accuracy_address)


                trainer = LoRAGaussSeidelTrainer(
                    LoRA_mode=LoRA_mode,
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=eval_dataset,
                    callbacks=[loss_logging_callback, accuracy_logging_callback,EvaluateAtEndOfEpochCallback()]  
                )
                trainer.train()
