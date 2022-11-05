import numpy as np

import torch.nn as nn 
import torch

from consts import device
from pipeline import make_teacher_model

def transfer_last_layer(teacher_model, num_classes=10):
    student_model = make_teacher_model()

    for i, stack in enumerate(teacher_model.stack):
        if i < len(teacher_model.stack) - 1:
            student_model.stack[i][0].weight = nn.Parameter(
                stack[0].weight.data.clone().detach().to(device), 
                requires_grad=True
            )
            student_model.stack[i][0].bias = nn.Parameter(
                stack[0].bias.data.clone().detach().to(device), 
                requires_grad=True
            )
        else:
            student_model.stack[i][0] = nn.Linear(
                in_features=stack[0].weight.shape[-1], 
                out_features=num_classes
            )
    student_model.to(device)

    return student_model

def freeze_layers(model, mask):
    for stack, freeze in zip(model.stack, mask):
        req_grad = freeze == False
        for param in stack[0].parameters():
            param.requires_grad = req_grad