
import copy
import numpy as np
from torchvision import datasets, transforms
import random
import torch
from torch import nn
import time

from utils.sampling import my_noniid
import argparse
from datetime import datetime
from models.Update import Final_test
from models.Nets import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")    
    parser.add_argument('--num_clients', type=int, default=10, help="number of clients")
    parser.add_argument('--num_users', type=int, default=100, help="number of users per clients")
    
    parser.add_argument('--sample_users', type=int, default=1000, help="number of users in federated learning C")
    
    parser.add_argument('--target_label', type=int, default=7, help="the poisoned label")
    
    parser.add_argument('--attack_ratio', type=float, default=0.0, help= "ratio of attacker in sampled users")
    parser.add_argument('--attack_mode', type=str, default="", choices=["poison", ""], help="type of attack")
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
    parser.add_argument('--seed', type=int, default=32, help='random seed')

    parser.add_argument('--shuffle', action='store_true', help='shuffle or not')
    parser.add_argument('--target_random', action='store_true', help='target label random or not')
    parser.add_argument('--noniid', type=float, default=0.4, help="non-iid rate")
    parser.add_argument('--scale', action='store_true', help='scale or not')
    parser.add_argument('--final_model', type=str, default='final.pt', help="name of final model")

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

    if(args.dataset == 'mnist'):
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        
        model_num = 1
        models = [CNN_Model().to(args.device) for i in range(model_num)]
        for i in range(model_num):
            path = args.final_model
            models[i].load_state_dict( torch.load(path,map_location=torch.device('cpu')))

        test_idxs = [i for i in range(54000,60000)]
        
        test_dataset = Final_test(args=args, dataset=dataset_train, idxs=test_idxs)
 
        acc_test, loss_test, acc_per_label, poison_acc = test_dataset.test(models)
        
        acc_per_label_avg = sum(acc_per_label)/len(acc_per_label)

        print( "Testing accuracy: {:.6f} loss: {:.6}".format(acc_test, loss_test))
        if args.test_label_acc:
            print( "Testing Label Acc: {}".format(acc_per_label) )
            print( "Testing Avg Label Acc : {}".format(acc_per_label_avg))
        if args.attack_mode=='poison':
            print( "Poison Acc: {}".format(poison_acc) )


    print("=== End ===")
