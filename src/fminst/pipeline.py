from operator import mod
from pathlib import Path 

from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
import IPython
from IPython.display import clear_output


from consts import batch_size, data_path, num_workers, device, use_colab, local_path, colab_path, teacher_blocks, student_blocks


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,), (0.5,))
    ])

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = FashionMNIST(
        root=data_path, train=True, download=True,
        transform=transform
    )

    test_data = FashionMNIST(
        root=data_path, train=False, download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        train_data, shuffle=False,
        batch_size=batch_size, num_workers=num_workers
    )

    test_dataloader = DataLoader(
        test_data, shuffle=False,
        batch_size=batch_size, num_workers=num_workers
    )

    return train_dataloader, test_dataloader


def train_loop(model, history, mask, dataloader, loss_fn, optimizer, noise_dist=None, noise_eps=0.0, batch_mod=50):

    size = 0
    train_loss, correct = 0, 0
    batches = 0

    for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
        X, y = X.to(device), y.to(device)
        if noise_dist is not None:
            if noise_dist == 'norm':
                X = X + torch.randn(*X.shape).to(device) * noise_eps
            elif noise_dist == 'uniform':
                X = X + (torch.rand(*X.shape).to(device) - 1/2) * noise_eps
            else:
                raise ValueError('bad noise distribution')
                
        pred = model(X) * mask
        mask_idx = torch.as_tensor([bool(mask[elem]) for elem in y])
        y, pred = y[mask_idx], pred[mask_idx]

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        size += len(y)
        batches += 1
        if batch % batch_mod == 0:
            test_loop(model, history, mask, dataloader, loss_fn, quiet=True)

    train_loss /= batches
    correct /= size

    history['train_loss'].append(train_loss)
    history['train_acc'].append(correct)

    return history


def antidistil_loop(teacher_model, student_model, lambdas, mask, dataloader, loss_fn, optimizer, scheduler=None, noise_dist=None, noise_eps=0.0):

    size = 0
    train_loss, correct = 0, 0
    batches = 0

    for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
        X, y = X.to(device), y.to(device)

        pred = student_model(X) * mask

        if noise_dist is not None:
            if noise_dist == 'norm':
                X = X + torch.randn(*X.shape).to(device) * noise_eps
            elif noise_dist == 'uniform':
                X = X + (torch.rand(*X.shape).to(device) - 1/2) * noise_eps
            else:
                raise ValueError('bad noise distribution')
                
        noise_pred = student_model(X) * mask
        mask_idx = torch.as_tensor([bool(mask[elem]) for elem in y])
        y, pred, noise_pred = y[mask_idx], pred[mask_idx], noise_pred[mask_idx]

        loss = loss_fn(pred, noise_pred, y, lambdas, teacher_model, student_model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        size += len(y)
        batches += 1

    if scheduler is not None:
        scheduler.step()

    train_loss /= batches
    correct /= size

    print(f'Train Loss: {train_loss}')
    print(f'Train Acc: {correct}')


def test_loop(model, history, mask, dataloader, loss_fn, quiet=False):
    size = 0
    test_loss, correct = 0, 0
    batches = 0
    if quiet:
        range_ = dataloader
    else:
        rang_ = tqdm(dataloader, leave=False, desc="Batch #")
    with torch.no_grad():
        for batch, (X, y) in enumerate(range_):
            X, y = X.to(device), y.to(device)

            pred = model(X) * mask
            mask_idx = torch.as_tensor([bool(mask[elem]) for elem in y])
            y, pred = y[mask_idx], pred[mask_idx]

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += len(y)
            batches += 1

    test_loss /= batches
    correct /= size

    history['val_loss'].append(test_loss)
    history['val_acc'].append(correct)
    if not quiet:
        print(
            f"Validation accuracy: {(100*correct):>0.1f}%, Validation loss: {test_loss:>8f} \n")
    return history


def get_path():
    if use_colab:
        from google.colab import drive
        drive.mount('/content/drive')
        p = Path(colab_path)
    else:
        p = Path(local_path)
    
    if not p.exists():
        p.mkdir(parents=True)
    return str(p)


class MLP(nn.Module):
    def __init__(self, blocks, in_features=28*28, n_classes=10,  bias=True):
        super().__init__()

        in_features = [in_features, *blocks]
        out_features = [*blocks, n_classes]
        
        self.flatten = nn.Flatten()

        self.stack = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_features[idx], out_features[idx], bias=bias),
                nn.ReLU(),
            )
            for idx in range(len(blocks) + 1)
        ])
        
        
    def forward(self, X):
        return self.stack(self.flatten(X))

def make_teacher_model(bias=True):
    return MLP(blocks=teacher_blocks, bias=bias).to(device)


def make_student_model(bias=True):
    return MLP(blocks=student_blocks, bias=bias).to(device)


def test_loop_noise(model, history, mask, dataloader, epses):
    
    batches = 0
    original_params = []
    for p in model.parameters():
        original_params.append(p.data * 1.0)
    if 'param_noise_acc' not in history:
        history['param_noise_acc'] = []

    history['param_noise_acc'].append([])
    with torch.no_grad():
        
        for eps in epses:
            correct = 0
            size = 0
            for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
                X, y = X.to(device), y.to(device)
                for p_old, p in zip(original_params, model.parameters()):
                    p.data *= 0
                    p.data += p_old + torch.randn(p.data.shape).to(device) * eps 
                    
                pred = model(X) * mask
                mask_idx = torch.as_tensor([bool(mask[elem]) for elem in y])
                y, pred = y[mask_idx], pred[mask_idx]

                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                size += len(y)
                batches += 1

            correct /= size
            history['param_noise_acc'][-1].append(correct)
    print ('Noise Accuracy', history['param_noise_acc'][-1])

    return history

def test_loop_fsgm(model, history, mask, dataloader, loss_fn, epses):
    
    batches = 0
    original_params = []
    for p in model.parameters():
        original_params.append(p.data * 1.0)
    if 'fsgm_noise_acc' not in history:
        history['fsgm_noise_acc'] = []

    history['fsgm_noise_acc'].append([])

    for eps in epses:
        correct = 0
        size = 0
        for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
            X, y = X.to(device), y.to(device)
            model.zero_grad()
            X.requires_grad = True 
            pred = model(X) * mask
            loss = loss_fn(pred, y)
            loss.backward()
            data_grad = X.grad.data 
            X2 = X + eps * torch.sign(data_grad)
            pred = model(X2) * mask 

            mask_idx = torch.as_tensor([bool(mask[elem]) for elem in y])
            y, pred = y[mask_idx], pred[mask_idx]

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += len(y)
            batches += 1

        correct /= size
        history['fsgm_noise_acc'][-1].append(correct)
    print ('FSGM Accuracy', history['fsgm_noise_acc'][-1])

    return history
    

def hessian_trace_reg(mlp, x, y, avg_num=1):
    def model_from_params(params):
        x_ = x
        real_params = list(mlp.parameters())
        offset = 0
        for i in range(len(real_params)//2):
            weight_real = real_params[2*i]
            bias_real = real_params[2*i+1]
            
            weight = params[offset:offset+np.prod(weight_real.shape)].reshape( -1, x_.shape[1])
            offset += np.prod(weight_real.shape)
            
            bias = params[offset:offset+bias_real.shape[0]]
            offset += bias.shape[0]
            
            x_ = (x_@weight.T) + bias
            x_ = torch.nn.functional.relu(x_)
        return criterion(x_, y)
        
    stack_params = torch.cat([p.flatten() for p in model.parameters()])
    res = []
    for _ in range(avg_num):
        random_vector = torch.rand(stack_params.shape[0])
        res.append(random_vector @ torch.autograd.functional.hvp(model_from_params, stack_params, random_vector)[1])
    return abs(sum(res)/avg_num)
    
