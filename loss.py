
import torch
import torch.nn as nn

def dice_loss(input, target, epsilon = 1e-7):

    size = input.size()
    length = len(size)

    if length == 3:
        input = input.view( -1)
        target = target.view( -1)
        
        intersection = torch.sum(target * input)
        union = torch.sum(target) + torch.sum(input)

        dice_error = ( (2 * intersection + epsilon) / (union + epsilon) )

    else:
        batch, _,_,_ = size
        input = input.view(batch, -1)
        target = target.view(batch, -1)

        intersection = torch.sum(target * input, dim=1)
        union = torch.sum(target, dim=1) + torch.sum(input, dim=1)

        dice_error = ( (2 * intersection + epsilon) / (union + epsilon) )

    return torch.mean(1 - dice_error)


bce_loss = nn.BCELoss()


def dice_bce(input, target, lambda_dice=0.5, lambda_ce=0.5):

    Dloss = dice_loss(input, target)
    # print(Dloss)

    CEloss = bce_loss(input, target.float())
    # print(CEloss)
    
    return lambda_dice * Dloss + lambda_ce * CEloss


