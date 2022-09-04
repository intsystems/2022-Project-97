import torch.nn as nn 
import torch

from consts import device
from pipeline import make_student_model


cross_entropy = nn.CrossEntropyLoss()


def get_weights(mode, weight_shape, bias_shape, student_model):
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

        weight, bias = get_weights(mode, weight_shape, bias_shape, student_model)

        weight[:weight_shape[0], :weight_shape[1]] = teacher_model.stack[i][0].weight
        student_model.stack[i][0].weight = nn.Parameter(weight)

        bias[:bias_shape[0]] = teacher_model.stack[i][0].bias
        student_model.stack[i][0].bias = nn.Parameter(bias)

    return student_model


def L2(teacher_model, student_model):
    norm = 0

    n = len(teacher_model.stack)
    for i in range(n):
        weight_shape = teacher_model.stack[i][0].weight.shape
        bias_shape = teacher_model.stack[i][0].bias.shape

        norm += torch.norm(student_model.stack[i][0].weight[:weight_shape[0], :weight_shape[1]] - teacher_model.stack[i][0].weight)
        norm += torch.norm(student_model.stack[i][0].bias[:bias_shape[0]] - teacher_model.stack[i][0].bias)

    return norm

def L4():
    return 0

def altidistill_loss(pred, noise_pred, y, lambdas, teacher_model, init_uniform=False):
    if init_uniform:
        student_model = simple_baseline_change_weights(teacher_model, 'uniform')
    else:
        student_model = make_student_model()

    loss = lambdas[0] * cross_entropy(pred, y) + lambdas[1] * L2(teacher_model, student_model) + lambdas[2] * cross_entropy(noise_pred, y) + lambdas[3] * L4()
    return loss