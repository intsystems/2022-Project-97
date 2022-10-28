import numpy as np

import torch.nn as nn 
import torch

from consts import device
from pipeline import make_student_model


cross_entropy = nn.CrossEntropyLoss()


def get_weights(mode, weight_shape, bias_shape, student_model, i):
    if mode == 'zero':
        weight = torch.zeros(weight_shape)
        bias = torch.zeros(bias_shape)
    elif mode == 'uniform':
        weight = student_model.stack[i][0].weight.clone()
        bias = student_model.stack[i][0].bias.clone()
    else:
        raise TypeError
    return weight, bias


def simple_baseline_change_weights(teacher_model, mode):
    student_model = make_student_model()
    
    if mode not in ['zero', 'uniform']:
        raise ValueError('bad mode')

    n = len(teacher_model.stack)
    for i in range(n):
        weight_shape = teacher_model.stack[i][0].weight.shape
        bias_shape = teacher_model.stack[i][0].bias.shape

        weight, bias = get_weights(mode, weight_shape, bias_shape, student_model, i)

        weight[:weight_shape[0], :weight_shape[1]] = teacher_model.stack[i][0].weight
        student_model.stack[i][0].weight = nn.Parameter(weight.to(device))

        bias[:bias_shape[0]] = teacher_model.stack[i][0].bias
        student_model.stack[i][0].bias = nn.Parameter(bias.to(device))

    return student_model


def L2(teacher_model, student_model):
    params = []

    n = len(teacher_model.stack)

    for i in range(n):
        weight_shape = teacher_model.stack[i][0].weight.shape
        bias_shape = teacher_model.stack[i][0].bias.shape

        params.append((student_model.stack[i][0].weight[:weight_shape[0], :weight_shape[1]] - teacher_model.stack[i][0].weight).flatten())
        params.append((student_model.stack[i][0].bias[:bias_shape[0]] - teacher_model.stack[i][0].bias).flatten())

    norm = torch.norm(torch.cat(params))

    return norm


def L4(mlp, x, y, criterion, avg_num=1):
    def model_from_params(params):
        x_ = x.view(x.shape[0], -1)
        real_params = list(mlp.parameters())
        offset = 0
        for i in range(len(real_params)//2):
            weight_real = real_params[2*i]
            bias_real = real_params[2*i+1]
            
            weight = params[offset:offset+np.prod(weight_real.shape)].reshape( -1, x_.shape[1])
            offset += np.prod(weight_real.shape)
            
            bias = params[offset:offset+bias_real.shape[0]]
            offset += bias.shape[0]
            
            x_ = (x_ @ weight.T) + bias
            x_ = torch.nn.functional.relu(x_)
        return criterion(x_, y)
        
    stack_params = torch.cat([p.flatten() for p in mlp.parameters()])
    res = []
    for _ in range(avg_num):
        random_vector = torch.rand(stack_params.shape[0]).to(device)
        res.append(random_vector @ torch.autograd.functional.hvp(model_from_params, stack_params, random_vector)[1])
    return abs(sum(res)/avg_num)


def altidistill_loss(pred, noise_pred, y, lambdas, teacher_model, student_model, X):
    loss = 0
    loss += lambdas[0] * cross_entropy(pred, y)
    loss += lambdas[1] * L2(teacher_model, student_model)
    loss += lambdas[2] * cross_entropy(noise_pred, y)
    loss += lambdas[3] * L4(student_model, X, y, cross_entropy) / 10000
    return loss