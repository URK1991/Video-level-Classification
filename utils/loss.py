import torch.nn.functional as nnf

def my_custom_loss(output, target, criterion):
    ce_loss = criterion(output, target)
    tm = []
    prob = nnf.softmax(output, dim=1)
    
    for i in range(len(prob)):
        mx = torch.max(prob[i])
        ind = int(torch.argmax(prob[i]))
        t_ind = int(torch.argmax(target[i]))
        re = 1 if t_ind != ind else 1 - mx
        tm.append(re)

    re_loss = sum(tm) / len(tm) if tm else 0  # Prevent division by zero
    return 0.5 * ce_loss + 0.5 * re_loss
