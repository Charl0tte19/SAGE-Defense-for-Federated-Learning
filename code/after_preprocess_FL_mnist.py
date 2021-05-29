#/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import random
import torch
import argparse
import time
from torchvision import datasets, transforms
from torch import nn

from utils.sampling import my_noniid
from datetime import datetime
from models.Update import LocalUpdate_0, LocalUpdate_1
from models.Fed import FedAvg
from models.test import test_img_poison
from models.Nets import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--total_users', type=int, default=1000, help="number of all users")
    parser.add_argument('--num_attacker', type=int, default=100, help="number of users per clients")
    parser.add_argument('--sample_users', type=int, default=100, help="number of users in federated learning C")
    
    parser.add_argument('--target_label', type=int, default=7, help="the poisoned label")
    
    parser.add_argument('--attack_ratio', type=float, default=0.0, help= "ratio of attacker in sampled users")
    parser.add_argument('--attack_mode', type=str, default="poison", choices=["poison", ""], help="type of attack")
    parser.add_argument('--aggregation', type=str, default="FedAvg", choices=["FedAvg"], help="name of aggregation method")
    parser.add_argument('--test_label_acc', action='store_true', help='obtain acc of each label and poinson acc')

    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=32, help='random seed (default: 1)')

    parser.add_argument('--shuffle', action='store_true', help='shuffle or not')
    parser.add_argument('--target_random', action='store_true', help='target label random or not')
    parser.add_argument('--model_path', type=str, default='./model.pt', help="path to save model")
    parser.add_argument('--noniid', type=float, default=0.4, help="non-iid rate")
    parser.add_argument('--scale', action='store_true', help='scale or not')   
    parser.add_argument('--pretrained_model', type=str, default='./model.pt', help="load pretrained model")
    parser.add_argument('--local_file', type=str, default='./test.txt', help="load local user file")
    parser.add_argument('--attacker_file', type=str, default='./test.txt', help="load attacker file")

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    
    dict_users, idxs_labels = my_noniid(dataset_train, args)

    torch.manual_seed(args.seed)

    net_glob = CNN_Model().to(args.device)
    path = args.pretrained_model
    net_glob.load_state_dict( torch.load(path) )

    print(net_glob)
    
    w_glob = net_glob.state_dict()

    loss_train_epoch = []
        

    if(args.attack_ratio <= 0.1):
        args.target_label = [7]
    elif(args.attack_ratio <= 0.2):
        args.target_label = [7,3]
    elif(args.attack_ratio <= 0.3):
        args.target_label = [7,3,5]
    elif(args.attack_ratio <= 0.4):
        args.target_label = [7,3,5,1]
    elif(args.attack_ratio <= 0.5):
        args.target_label = [7,3,5,1,9]


    np.random.seed(args.seed)
    
    with open(args.local_file,'r') as f:
        locals = eval(f.read())
    
    print('locals: ',locals)

    args.total_users = len(locals)

    if(args.attack_mode == "poison"):
        all_attacker = []
        print('target_label:',args.target_label)
        print("")

    else:
        exit()
    

    if(args.attack_mode == "poison"):
        with open(args.attacker_file,'r') as f:
            attacker_idxs = eval(f.read())
        
        for a in attacker_idxs:
            if(a in locals):
                all_attacker.append(a)

    if(args.attack_mode=='poison'):
            print("number of attacker: ",len(all_attacker))
            print("all attacker: ", all_attacker) 
            print("")

    pre_norm = 0
    total_time = 0
    
    for round in range(args.epochs):
        w_locals, loss_locals = [], []
        attacker_idxs = []
        idxs_users = np.random.choice(locals, args.sample_users, replace=False)
        
        print("Randomly selected {}/{} users for federated learning. {}".format(args.sample_users, args.total_users, datetime.now().strftime("%H:%M:%S")))

        count = 0
        start_time = time.time()
        for idx in idxs_users:
            
            if(args.attack_mode == "poison"):
                
                local = LocalUpdate_1(args=args, dataset=dataset_train, idxs=idxs_labels[0][dict_users[idx]], user_idx=idx, attack_idxs=all_attacker,round=round)
                
                w, loss, attack_flag, count_1 = local.train(net=copy.deepcopy(net_glob).to(args.device))

                count += count_1

                if(attack_flag):
                    attacker_idxs.append(idx)

                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))


        print( "{}/{} are attackers with {} attack".format(len(attacker_idxs), args.sample_users, args.attack_mode) )

        print(" count(target label): ",count)

        user_sizes = np.array([ len(dict_users[idx]) for idx in idxs_users ])
        user_weights = user_sizes / float(sum(user_sizes))
        if args.aggregation == "FedAvg":
            w_glob = FedAvg(w_locals, user_weights)

        if(round==0):
            pre_net_glob = w_glob

        pre_net_glob = copy.deepcopy(pre_net_glob)

        net_glob.load_state_dict(w_glob)
        
        loss_avg = np.sum(loss_locals * user_weights)

        print('=== Round {:3d}, Average loss {:.6f} ==='.format(round, loss_avg))
        print("{} users; time {}".format(len(idxs_users), datetime.now().strftime("%H:%M:%S")) )

        acc_test, loss_test, acc_per_label, poison_acc = test_img_poison(copy.deepcopy(net_glob).to(args.device), dataset_test, args)
        acc_per_label_avg = sum(acc_per_label)/len(acc_per_label)

        print( " Testing accuracy: {:.6f} loss: {:.6}".format(acc_test, loss_test))
        if args.test_label_acc:
            print( " Testing Label Acc: {}".format(acc_per_label) )
            print( " Testing Avg Label Acc : {}".format(acc_per_label_avg))
        if args.attack_mode=='poison':
            print( " Poison Acc: {}".format(poison_acc) )
            print( "======")


        pre_norm_net = torch.tensor([])
        norm_net = torch.tensor([])

        for k in net_glob.state_dict().values():
            norm_net = torch.cat((norm_net,torch.flatten(k).cpu()))
            
        for k in pre_net_glob.values():
            pre_norm_net = torch.cat((pre_norm_net,torch.flatten(k).cpu()))
            
        norm = torch.norm(norm_net-pre_norm_net)
        print("Model_Weight_Norm :",norm)
        print("Model_weight_Slope:",norm-pre_norm)
        pre_norm = norm
        
        
        pre_net_glob = copy.deepcopy(w_glob)
        
        print("Test end {}".format(datetime.now().strftime("%H:%M:%S")))
        print("-------------------------------------------------------------------------")
        print("")

        end_time = time.time()
        round_time = end_time - start_time
        total_time += round_time
        print("round_time: ",round_time)
        print("")
        loss_train_epoch.append(loss_avg)

    
    torch.save(net_glob.state_dict(), args.model_path)

    print("total_time: ",total_time)
    print("=== End ===")
