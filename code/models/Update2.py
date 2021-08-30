'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/Update.py
'''

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import torch.nn.functional as F


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # image: torch.Size([1, 28, 28]), torch.float32; label: int
        return image, label

class LocalUpdate_0(object):

    def __init__(self, args, dataset=None, idxs=None, user_idx=None, attack_idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)

        self.user_idx = user_idx

        self.attack_idxs = attack_idxs

        self.poison_attack = False

        self.attacker_flag = False

    def split(self, net):
        
        random.seed(self.args.seed)
        attack_or_not = random.choices([1,0],k=1,weights=[self.attack_idxs[2],1-self.attack_idxs[2]])

        answer = set([0,1,2,3,4,5,6,7,8,9])

        answer = list(answer - set(self.args.target_label))
        
        count = 0
        label_count = 0
        a = 0
            
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
        
            if (self.args.attack_mode == 'poison'):
                
                for label_idx in range(len(labels)):
                    if(labels[label_idx] in self.args.target_label):
                            label_count += 1

        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            if(self.args.attack_mode == 'poison'):
                if((self.args.dataset=="mnist" or self.args.dataset=='fmnist') and label_count >= int(54000//self.args.total_users*self.args.noniid)):
                    count = 1 

                for label_idx in range(len(labels)):
                    if (count==1 and labels[label_idx] in self.args.target_label) and (self.attack_idxs[0] > self.attack_idxs[1]) and attack_or_not[0]:
                        self.attacker_flag = True
                        if(self.args.target_random == True):
                            if(a==0):
                                print(self.user_idx)
                                a=1

                        else:
                            pass

        return self.attacker_flag



class LocalUpdate_1(object):
    
    def __init__(self, args, dataset=None, idxs=None, user_idx=None, attack_idxs=None, round=1):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset
        
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.user_idx = user_idx

        self.attack_idxs = attack_idxs

        self.poison_attack = False

        self.attacker_flag = False

        self.round = round

    def train(self, net):
        net.train()
        
        shared_weights = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        labels_count = []

        answer = set([0,1,2,3,4,5,6,7,8,9])

        answer = list(answer - set(self.args.target_label))

        count = 0

        label_count = 0
        a = -1

        for iter in range(self.args.local_ep):
            batch_loss = []
            
            if(a == 0):
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
            
                    if (self.args.attack_mode == 'poison'):
                        for label_idx in range(len(labels)):
                            if(labels[label_idx] in self.args.target_label):
                                label_count += 1
                print('label_count: ',label_count)
                a = 1

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                
                for label_idx in range(len(labels)):
                    
                    if(self.args.attack_mode == 'poison') and (labels[label_idx] in self.args.target_label):
                        count = 1
                        
                    if (self.args.attack_mode == 'poison') and (labels[label_idx] in self.args.target_label)  and (self.user_idx in self.attack_idxs):
                        self.attacker_flag = True
                            
                        if(self.args.target_random == True):
                            labels[label_idx] = random.choices(answer,k=1)[0]
                        else:    
                            labels[label_idx] = int(self.args.target_label + 1)%10

                    else:
                        pass    
                
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()

                log_probs = net(images)
                
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            
            if self.args.verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                        iter, epoch_loss[iter]))

        
        # model replacement
        trained_weights = copy.deepcopy(net.state_dict())


        if(self.args.scale==True):
            scale_up = 20
        else:    
            scale_up = 1
        if (self.args.attack_mode == "poison") and self.attacker_flag:

            attack_weights = copy.deepcopy(shared_weights)
            
            for key in shared_weights.keys():
                difference =  trained_weights[key] - shared_weights[key]
                attack_weights[key] += scale_up * difference
            
            return attack_weights, sum(epoch_loss) / len(epoch_loss), self.attacker_flag, count

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.attacker_flag, count




class Final_test(object):
    
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.dataset = dataset
        
        self.data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def test(self,net_g):
    

        for i in range(len(net_g)):
            net_g[i].eval()
        # testing
        test_loss = 0
        if self.args.dataset == "mnist":
            correct  = torch.tensor([0.0] * 10)
            gold_all = torch.tensor([0.0] * 10)
        elif self.args.dataset == "fmnist":
            correct  = torch.tensor([0.0] * 10)
            gold_all = torch.tensor([0.0] * 10)
        else:
            print("Unknown dataset")
            exit(0)

        poison_correct = 0.0

        l = len(self.data_loader)
        print('test data_loader(per batch size):',l)

        log_probs = [None for i in range(len(net_g))]
        test_loss = [0 for i in range(len(net_g))]
        y_pred = [None for i in range(len(net_g))]


        for idx, (data, target) in enumerate(self.data_loader):
        
            if self.args.gpu != -1:
                data, target = data.to(self.args.device), target.to(self.args.device)

            for m in range(len(net_g)):
                
                log_probs[m] = net_g[m](data)
          
                test_loss[m] += F.cross_entropy(log_probs[m], target, reduction='sum').item()
                
                y_pred[m] = log_probs[m].data.max(1, keepdim=True)[1]
                
                y_pred[m] = y_pred[m].squeeze(1)
                
                
            answer = [None for i in range(len(net_g))]
            
            for i in range(len(y_pred[0])):
                for m in range(len(net_g)):
                    answer[m] = y_pred[m][i]
                maxlabel = max(answer,key=answer.count)
                y_pred[0][i] = maxlabel
            
            y_gold = target.data.view_as(y_pred[0])

            for pred_idx in range(len(y_pred[0])):
                
                gold_all[ y_gold[pred_idx] ] += 1
            
                if y_pred[0][pred_idx] == y_gold[pred_idx]:
                    correct[y_pred[0][pred_idx]] += 1
            
                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.1:
                    if(self.args.target_random == True):
                        if int(y_pred[0][pred_idx]) != 2 and int(y_gold[pred_idx]) == 2:  # poison attack
                            poison_correct += 1
                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.2:
                    if(self.args.target_random == True):
                        if int(y_pred[0][pred_idx]) != 2 and int(y_gold[pred_idx]) == 2:  # poison attack
                            poison_correct += 1
                        if int(y_pred[0][pred_idx]) != 4 and int(y_gold[pred_idx]) == 4:  # poison attack
                            poison_correct += 1
                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.3:
                    if(self.args.target_random == True):
                        if int(y_pred[0][pred_idx]) != 2 and int(y_gold[pred_idx]) == 2:  # poison attack
                            poison_correct += 1
                        if int(y_pred[0][pred_idx]) != 4 and int(y_gold[pred_idx]) == 4:  # poison attack
                            poison_correct += 1
                        if int(y_pred[0][pred_idx]) != 0 and int(y_gold[pred_idx]) == 0:  # poison attack
                            poison_correct += 1
                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.4:
                    if(self.args.target_random == True):
                        if int(y_pred[pred_idx]) != 2 and int(y_gold[pred_idx]) == 2:  # poison attack
                            poison_correct += 1
                        if int(y_pred[pred_idx]) != 4 and int(y_gold[pred_idx]) == 4:  # poison attack
                            poison_correct += 1
                        if int(y_pred[pred_idx]) != 0 and int(y_gold[pred_idx]) == 0:  # poison attack
                            poison_correct += 1    
                        if int(y_pred[pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                            poison_correct += 1         

                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.5:
                    if(self.args.target_random == True):
                        if int(y_pred[pred_idx]) != 2 and int(y_gold[pred_idx]) == 2:  # poison attack
                            poison_correct += 1
                        if int(y_pred[pred_idx]) != 4 and int(y_gold[pred_idx]) == 4:  # poison attack
                            poison_correct += 1
                        if int(y_pred[pred_idx]) != 0 and int(y_gold[pred_idx]) == 0:  # poison attack
                            poison_correct += 1    
                        if int(y_pred[pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                            poison_correct += 1            
                        if int(y_pred[pred_idx]) != 6 and int(y_gold[pred_idx]) == 6:  # poison attack
                            poison_correct += 1 


        for i in range(len(net_g)):
            test_loss[i] /= len(self.data_loader.dataset)
        
        test_loss = sum(test_loss)/len(test_loss)

        accuracy = (sum(correct) / sum(gold_all)).item()
    
        acc_per_label = correct / gold_all

        poison_acc = 0

        if(self.args.attack_mode == 'poison' and self.args.attack_ratio <= 0.1):
            poison_acc = poison_correct/gold_all[self.args.target_label].item()
        elif(self.args.attack_mode == 'poison'  and self.args.attack_ratio <= 0.2 ):
            poison_acc = poison_correct/(gold_all[2].item()+gold_all[4].item())
        elif(self.args.attack_mode == 'poison'  and self.args.attack_ratio <= 0.3 ):
            poison_acc = poison_correct/(gold_all[2].item()+gold_all[4].item()+gold_all[0].item())
        elif(self.args.attack_mode == 'poison'  and self.args.attack_ratio <= 0.4 ):
            poison_acc = poison_correct/(gold_all[2].item()+gold_all[4].item()+gold_all[0].item()+gold_all[3].item())
        elif(self.args.attack_mode == 'poison'  and self.args.attack_ratio <= 0.5 ):
            poison_acc = poison_correct/(gold_all[2].item()+gold_all[4].item()+gold_all[0].item()+gold_all[3].item()+gold_all[6].item())


        return accuracy, test_loss, acc_per_label.tolist(), poison_acc
