'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/test.py
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

gpu = 0
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and gpu != -1 else 'cpu')

def test_img_poison(net_g, datatest, args):

    net_g.eval()
    test_loss = 0
    if args.dataset == "mnist":
        correct  = torch.tensor([0.0] * 10)
        gold_all = torch.tensor([0.0] * 10)
    else:
        print("Unknown dataset")
        exit(0)

    poison_correct = 0.0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    
    l = len(data_loader)
    print(' test data_loader(per batch size):',l)
    
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        
        log_probs = net_g(data)
        
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        
        y_pred = log_probs.data.max(1, keepdim=True)[1]

        y_gold = target.data.view_as(y_pred).squeeze(1)
        y_pred = y_pred.squeeze(1)


        for pred_idx in range(len(y_pred)):
            
            gold_all[ y_gold[pred_idx] ] += 1
            
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1
            
            elif args.attack_mode == 'poison' and args.attack_ratio<=0.1:
                if(args.target_random == True):
                    if int(y_pred[pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                        poison_correct += 1

            elif args.attack_mode == 'poison' and args.attack_ratio<=0.2:
                if(args.target_random == True):
                    if int(y_pred[pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                        poison_correct += 1
                    if int(y_pred[pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                        poison_correct += 1

            elif args.attack_mode == 'poison' and args.attack_ratio<=0.3:
                if(args.target_random == True):
                    if int(y_pred[pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                        poison_correct += 1
                    if int(y_pred[pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                        poison_correct += 1
                    if int(y_pred[pred_idx]) != 5 and int(y_gold[pred_idx]) == 5:  # poison attack
                        poison_correct += 1

            elif args.attack_mode == 'poison' and args.attack_ratio<=0.4:
                if(args.target_random == True):
                    if int(y_pred[pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                        poison_correct += 1
                    if int(y_pred[pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                        poison_correct += 1
                    if int(y_pred[pred_idx]) != 5 and int(y_gold[pred_idx]) == 5:  # poison attack
                        poison_correct += 1    
                    if int(y_pred[pred_idx]) != 1 and int(y_gold[pred_idx]) == 1:  # poison attack
                        poison_correct += 1         

            elif args.attack_mode == 'poison' and args.attack_ratio<=0.5:
                if(args.target_random == True):
                    if int(y_pred[pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                        poison_correct += 1
                    if int(y_pred[pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                        poison_correct += 1
                    if int(y_pred[pred_idx]) != 5 and int(y_gold[pred_idx]) == 5:  # poison attack
                        poison_correct += 1    
                    if int(y_pred[pred_idx]) != 1 and int(y_gold[pred_idx]) == 1:  # poison attack
                        poison_correct += 1            
                    if int(y_pred[pred_idx]) != 9 and int(y_gold[pred_idx]) == 9:  # poison attack
                        poison_correct += 1 

    test_loss /= len(data_loader.dataset)

    accuracy = (sum(correct) / sum(gold_all)).item()
    
    acc_per_label = correct / gold_all

    poison_acc = 0

    if(args.attack_mode == 'poison' and args.attack_ratio <= 0.1):
        poison_acc = poison_correct/gold_all[args.target_label].item()
    elif(args.attack_mode == 'poison'  and args.attack_ratio <= 0.2 ):
        poison_acc = poison_correct/(gold_all[7].item()+gold_all[3].item())
    elif(args.attack_mode == 'poison'  and args.attack_ratio <= 0.3 ):
        poison_acc = poison_correct/(gold_all[7].item()+gold_all[3].item()+gold_all[5].item())
    elif(args.attack_mode == 'poison'  and args.attack_ratio <= 0.4 ):
        poison_acc = poison_correct/(gold_all[7].item()+gold_all[3].item()+gold_all[5].item()+gold_all[1].item())
    elif(args.attack_mode == 'poison'  and args.attack_ratio <= 0.5 ):
        poison_acc = poison_correct/(gold_all[7].item()+gold_all[3].item()+gold_all[5].item()+gold_all[1].item()+gold_all[9].item())

    return accuracy, test_loss, acc_per_label.tolist(), poison_acc





