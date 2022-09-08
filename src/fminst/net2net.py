from turtle import st
import torch 
import random
from pipeline import MLP, make_student_model, make_teacher_model
from consts import teacher_blocks, student_blocks, device

def mlp_to_unbiased_mlp(layers1, layers2):
    for l1, l2 in zip(layers1, layers2):
        w1 = l1.weight.data.T
        w2 = l2.weight.data.T.clone()
        
        w2[:-1, :-1] = w1
        w2[-1, :-1] = l1.bias.view(1, -1)
        w2[:,-1] = 0
        w2[-1,-1]=1
        
        l2.weight.data *= 0
        l2.weight.data+=w2.T

def net2net_step(Wt1, Wt2,  inp, teacher_hidden, student_hidden, out_num, debug=False, use_bias=False):
    print ('inp', inp)
    print ('t hid', teacher_hidden)
    print ('s hid', student_hidden)
    print ('out_num', out_num)
    mapping_num = [1]*teacher_hidden
    mapping = {}

    Ws1 = torch.randn(inp,student_hidden)
    Ws2 = torch.randn(student_hidden,out_num)
    
    if use_bias:
        Ws1[:,:teacher_hidden] = Wt1[:,:teacher_hidden]
    else:
        Ws1[:, :teacher_hidden-1] = Wt1[:, :teacher_hidden-1]
        Ws1[:, -1] = Wt1[:, -1]
    
    if use_bias:
        range_ = range(teacher_hidden, student_hidden)
    else:
        range_ = range(teacher_hidden-1, student_hidden-1)
    for i in range_:
        if debug:
            id = teacher_hidden-1
        else:
            if use_bias:
                max_ = teacher_hidden -1 
            else:
                max_ = teacher_hidden -2 
            id = random.randint(0, max_)
            
        mapping_num[id]+=1
        mapping[i] = id
        Ws1[:,i] = Wt1[:,id]
    
    if use_bias:
        Ws2[:teacher_hidden] = Wt2[:teacher_hidden]
    else:
        Ws2[:teacher_hidden-1] = Wt2[:teacher_hidden-1]
        Ws2[-1] = Wt2[-1]
        
    if use_bias:
        range_ = range(student_hidden)
    else:
        range_ = range(student_hidden-1)
        
    for i in range_:
        id = mapping.get(i, i)
        num = mapping_num[id] 
        Ws2[i] = Wt2[id]/(num)
    return Ws1, Ws2

def net2net_mlp(mlp1, mlp2, use_bias=False):
    old_Ws2 = None
    new_matrices = []
    mlp2_weight_shapes = [w.weight.T.shape for w in mlp2]
    for i in range(len(mlp1)-1):
        if old_Ws2 is None:
            Wt1 = mlp1[i].weight.T
            Wt2 = mlp1[i+1].weight.T
        else:
            Wt1 = old_Ws2
            Wt2 = mlp1[i+1].weight.T
        #print (mlp1[i].weight.shape, mlp2[i].weight.shape)
        Ws1, Ws2 = net2net_step(Wt1, Wt2,  Wt1.shape[0],
                                Wt1.shape[1],
                                mlp2_weight_shapes[i][1], 
                                Wt2.shape[1], use_bias=use_bias)
        mlp2[i].weight.data = Ws1.T
        mlp2[i+1].weight.data = Ws2.T
        old_Ws2 = Ws2
        
        

def unbiased_mlp_to_mlp(layers1, layers2):
    for l1, l2 in zip(layers1, layers2):
        if isinstance(l1, torch.nn.ReLU):
            continue
            
        #print (l1, l2)\        
        w1 = l1.weight.data.T
        w2 = l2.weight.data.T.clone()
        
        w2 = w1[:-1, :-1]
        
        l2.weight.data *= 0      
        l2.weight.data+=w2.T
        l2.bias.data *= 0
        l2.bias.data += w1[-1, :-1]


def net2net_antidistil(teacher):
    teacher =teacher.to('cpu')
    student = make_student_model().to('cpu')
    unbiased_teacher = MLP(in_features=28*28+1, blocks= [t+1 for t in teacher_blocks], n_classes=11, bias=False)
    mlp_to_unbiased_mlp([l[0] for l in teacher.stack], [l[0] for l in unbiased_teacher.stack])
    
    unbiased_student = MLP(in_features=28*28+1, blocks= [t+1 for t in student_blocks], n_classes=11, bias=False)
    
    net2net_mlp([l[0] for l in unbiased_teacher.stack], [l[0] for l in unbiased_student.stack])

    unbiased_mlp_to_mlp([l[0] for l in unbiased_student.stack], [l[0] for l in student.stack])
    
    return student.to(device)
    
if __name__=='__main__':    
    x = torch.randn(4, 28*28)
    net2net_antidistil(teacher, student)
    print (teacher(x) - student(x))
    
