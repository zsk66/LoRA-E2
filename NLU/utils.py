import torch
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
from peft import get_peft_model, LoraConfig
import os
import math
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer
from peft.utils.integrations import gather_params_ctx




def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)




def stable_init(self, adapter_name):
    r = self.r[adapter_name]
    in_features = self.in_features
    out_features = self.out_features

    fan_in = in_features
    std = math.sqrt(2.0 / (fan_in ** 0.75))
    with torch.no_grad():
        self.lora_A[adapter_name].weight.data.normal_(mean=0.0, std=std)

    self.lora_B[adapter_name].weight.data.zero_()

    if adapter_name in self.lora_bias and self.lora_B[adapter_name].bias is not None:
        self.lora_B[adapter_name].bias.data.zero_()



def kaiming_init(self, adapter_name):
    r = self.r[adapter_name]
    in_features = self.in_features
    out_features = self.out_features

    
    fan_in = in_features
    std = math.sqrt(2.0 / (fan_in))
    with torch.no_grad():
        self.lora_A[adapter_name].weight.data.normal_(mean=0.0, std=std)

    self.lora_B[adapter_name].weight.data.zero_()

    if adapter_name in self.lora_bias and self.lora_B[adapter_name].bias is not None:
        self.lora_B[adapter_name].bias.data.zero_()


def new_update_layer(
    self,
    adapter_name,
    r,
    lora_alpha,
    lora_dropout,
    init_lora_weights,
    use_rslora,
    use_dora: bool = False,
    lora_bias: bool = False,
):
    if r <= 0:
        raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha

    if lora_dropout > 0.0:
        lora_dropout_layer = nn.Dropout(p=lora_dropout)
    else:
        lora_dropout_layer = nn.Identity()
    
    self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

    self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
    self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
    self.lora_bias[adapter_name] = lora_bias

    if use_rslora:
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
    else:
        self.scaling[adapter_name] = lora_alpha / r

    if init_lora_weights == "my_init":
        self.init_A_B_in_my_way(adapter_name)
    elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
        with gather_params_ctx(self.get_base_layer().weight):
            self.pissa_init(adapter_name, init_lora_weights)
    elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
        with gather_params_ctx(self.get_base_layer().weight):
            self.olora_init(adapter_name)
    elif init_lora_weights == "loftq":
        with gather_params_ctx(self.get_base_layer().weight):
            self.loftq_init(adapter_name)
    elif init_lora_weights == "eva":
        nn.init.zeros_(self.lora_B[adapter_name].weight)
    elif init_lora_weights:
        self.reset_lora_parameters(adapter_name, init_lora_weights)

    self._move_adapter_to_device_of_base_layer(adapter_name)

    if use_dora:
        self.dora_init(adapter_name)
        self.use_dora[adapter_name] = True
    else:
        self.use_dora[adapter_name] = False

    self.set_adapter(self.active_adapters)