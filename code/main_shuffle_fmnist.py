'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/main_regression.py
'''

import copy
import numpy as np
import argparse
import random
import torch
import time

from torchvision import datasets, transforms
from torch import nn

from utils.sampling import my_noniid
from datetime import datetime
from models.Update2 import LocalUpdate_0, LocalUpdate_1
from models.Fed import FedAvg
from models.test2 import test_img_poison
from models.Nets import *           


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")    
    parser.add_argument('--num_clients', type=int, default=10, help="number of clients")
    parser.add_argument('--num_users', type=int, default=100, help="number of users per clients")
    parser.add_argument('--total_users', type=int, default=1000, help="number of total users")
    
    
    parser.add_argument('--target_label', type=int, default=7, help="the poisoned label")
    parser.add_argument('--attack_ratio', type=float, default=0.0, help= "ratio of attacker in sampled users")
    parser.add_argument('--attack_mode', type=str, default="", choices=["poison", ""], help="type of attack")
    parser.add_argument('--aggregation', type=str, default="FedAvg", choices=["FedAvg"], help="name of aggregation method")
    parser.add_argument('--test_label_acc', action='store_true', help='obtain acc of each label and poinson acc')
    
    
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")


    parser.add_argument('--dataset', type=str, default='fmnist', choices=["fmnist"], help="name of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=32, help='random seed')

    parser.add_argument('--shuffle', action='store_true', help='shuffle or not')
    parser.add_argument('--target_random', action='store_true', help='target label changes to random label or not')
    parser.add_argument('--model_path', type=str, default='./model', help="path to save model")   
    parser.add_argument('--noniid', type=float, default=0.4, help="non-iid rate")
    parser.add_argument('--scale', action='store_true', help='scale or not')    
    
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

    if(args.dataset == 'fmnist'):
        print('==> Preparing data..')

        transform = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST(
            root='../data/fmnist', train=True, download=True, transform=transform)
        
        dataset_test = datasets.FashionMNIST(
            root='../data/fmnist', train=False, download=True, transform=transform)
        
        args.total_users = args.num_clients*args.num_users
        dict_users, idxs_labels = my_noniid(dataset_train, args)


        torch.manual_seed(args.seed)                     
        net_glob = Network().to(args.device)

    else:
        print('just for fmnist')
        exit()
    

    print(net_glob)
    net_glob.train()

    w_glob = net_glob.state_dict()

    loss_train_epoch = []
        
    attacker_num  = int( args.attack_ratio * args.total_users)
    all_attacker = []
    attacker_count = 0
    attack_or_not = 1

    attack_set = [attacker_num, attacker_count, attack_or_not]


    if(args.attack_ratio <= 0.1):
        args.target_label = [2]
    elif(args.attack_ratio <= 0.2):
        args.target_label = [2,4]
    elif(args.attack_ratio <= 0.3):
        args.target_label = [2,4,0]
    elif(args.attack_ratio <= 0.4):
        args.target_label = [2,4,0,3]
    elif(args.attack_ratio <= 0.5):
        args.target_label = [2,4,0,3,6]
        
    np.random.seed(args.seed)                  
    if(args.attack_mode == "poison"):
        print('target_label:',args.target_label)
        print("")

    else:
        all_attacker = np.random.choice(range(args.total_users), attacker_num, replace=False)
        print("all attacker: ", all_attacker) 
        print("")
        
    
    all_users = [i for i in range(args.total_users)]
    local_users = [i for i in range(args.num_clients)]
    attacked = [i for i in range(args.num_clients)]

    rec_locals = [i for i in range(args.num_clients)]

    net_glob_backup = [ copy.deepcopy(net_glob) for _ in range(args.num_clients)]

    # choose attackers
    if(args.attack_mode == "poison"):
        attacker_idxs = []
        idxs_users = np.random.choice(range(args.total_users), args.total_users, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_0(args=args, dataset=dataset_train, idxs=idxs_labels[0][dict_users[idx]], user_idx=idx, attack_idxs=attack_set)
            attack_flag = local.split(net=copy.deepcopy(net_glob).to(args.device))
            if(attack_flag):
                attacker_idxs.append(idx)
                attack_set[1] += 1

                if(attack_set[2]>1):
                    attack_set[2] -= 1 / (100*args.attack_ratio)
            
            if(attack_set[1]==attack_set[0]):
                break
  
    if(args.attack_mode=='poison'):
        print("number of attacker: ",attack_set[1])
        all_attacker.extend(attacker_idxs)         
        print("all attacker: ", all_attacker)   
        print("")                          
    

    for i in range(args.num_clients):
        local_users[i] = set(np.random.choice(all_users, args.num_users, replace=False))
        all_users = list(set(all_users) - local_users[i])


    good = []
    normal = [i for i in range(args.num_clients)]
    bad = []

    off = 0
    best_std = 0
    first_save = 0
    client_in_good = [0 for i in range(args.num_clients)]
    target = int(0.3*args.num_clients)
    
    pre_norm = 0
    step1_finish = 0
    clients_norm = [0 for i in range(args.num_clients)]

    total_time = 0
    pre_acc_avg = 0

    acc_rec = [0 for i in range(args.num_clients)]
    acc_per_label_avg = [0 for i in range(args.num_clients)]
    acc_per_label_std = [0 for i in range(args.num_clients)]


    for round in range(args.epochs):

        w_locals, loss_locals = [[] for i in range(args.num_clients)], [[] for i in range(args.num_clients)]
        attacker_idxs = [[] for i in range(args.num_clients) ]
        
        shuffle_in_normal = []
        shuffle_in_good = []
        shuffle_in_bad = []
        
        acc_rec_normal = []
        acc_rec_good = []
        acc_rec_bad = []

        local_ep_time = []
        global_test_time = 0
        

        print("good: ",good)
        print("normal: ",normal)
        print("bad: ",bad)
        
        # training
        for client in range(args.num_clients):

            attack_set[2] = 1

            idxs_users = local_users[client]
            
            count = 0

            start_time = time.time()

            for idx in idxs_users:
                
                if(args.attack_mode == "poison"):
                    local = LocalUpdate_1(args=args, dataset=dataset_train, idxs=idxs_labels[0][dict_users[idx]], user_idx=idx, attack_idxs=all_attacker,round=round)
                    w, loss, attack_flag, count_1 = local.train(net=copy.deepcopy(net_glob_backup[client]).to(args.device))
                    
                    count += count_1
                    
                    if(attack_flag):
                        attacker_idxs[client].append(idx)
                
                else:
                    local = LocalUpdate_1(args=args, dataset=dataset_train, idxs=idxs_labels[0][dict_users[idx]], user_idx=idx, attack_idxs=all_attacker, round=round)
                    w, loss, attack_flag, count_1 = local.train(net=net_glob_backup[client].to(args.device))

                w_locals[client].append(copy.deepcopy(w))
                loss_locals[client].append(copy.deepcopy(loss))                                                     

            
            print("Client {}".format(client))
            print(" {}/{} are attackers with {} attack ".format(len(attacker_idxs[client]), len(local_users[client]), args.attack_mode))

            print(" count(target label): ",count)

            end_time = time.time()

            local_ep_time.append(end_time - start_time)

            start_time = time.time()

            user_sizes = np.array([ len(dict_users[idx]) for idx in idxs_users ])
            user_weights = user_sizes / float(sum(user_sizes))

            if args.aggregation == "FedAvg":
                w_glob = FedAvg(w_locals[client], user_weights)
            else:
                print('no other aggregation method.')
                exit()
        
            pre_net_glob = copy.deepcopy(net_glob_backup[client].state_dict())
            # copy weight to net_glob
            net_glob_backup[client].load_state_dict(w_glob)

            # print loss
            loss_avg = np.sum(loss_locals[client] * user_weights)


            print('=== Round {:3d}, Average loss {:.6f} ==='.format(round, loss_avg))
            print(" {} users; time {}".format(len(idxs_users), datetime.now().strftime("%H:%M:%S")) )

            acc_test, loss_test, acc_per_label, poison_acc = test_img_poison(copy.deepcopy(net_glob_backup[client]).to(args.device), dataset_test, args)
            acc_rec[client] = acc_test
            acc_per_label_avg[client] = sum(acc_per_label)/len(acc_per_label)
            acc_per_label_std[client] = np.std(acc_per_label)


            print( " Testing accuracy: {} loss: {:.6}".format(acc_test, loss_test))
            if args.test_label_acc:
                print( " Testing Label Acc: {}".format(acc_per_label) )
                print( " Testing Avg Label Acc : {}".format(acc_per_label_avg[client]))
                print( " Testing Std Label Acc : {}".format(acc_per_label_std[client]))
            if args.attack_mode=='poison':
                print( " Poison Acc: {}".format(poison_acc) )
                     
            
            if(client in normal):
                acc_rec_normal.append(acc_per_label_avg[client])

            if(client in good):
                acc_rec_good.append(acc_per_label_avg[client])

            if(client in bad):
                acc_rec_bad.append(acc_per_label_avg[client])
            
                     
            print( "======")
            print("")
        
            # finish an iteration for all master models

        print("Test end {}".format(datetime.now().strftime("%H:%M:%S")))
        end_time = time.time()
        global_test_time += (end_time - start_time)
        
        print("-------------------------------------------------------------------------")
        print("")

        acc_list = list(enumerate(acc_rec))
        acc_list.sort(key=lambda tup:tup[1],reverse=True)
        print('acc_list: ',acc_list)
        
        max_local_ep_time = local_ep_time[0]
        for k in range(1,args.num_clients):
            if(local_ep_time[k]>local_ep_time[k-1]):
                max_local_ep_time = local_ep_time[k]
        
        start_time = time.time()

        
        if(round == 0)and(args.attack_mode=='poison'):
            
            if(attack_set[0] > attack_set[1]):
                attack_set[1] = attack_set[0]+1

        acc_avg_nomral = 0
        acc_avg_good = 0
        acc_avg_bad = 0

        if(args.shuffle):
            regroup_on = abs(sum(acc_per_label_avg)/len(acc_per_label_avg) - pre_acc_avg)
            acc_avg = sum(acc_per_label_avg)/len(acc_per_label_avg)
            pre_acc_avg = acc_avg
            print('acc_avg_difference: ',regroup_on)

        
        if(args.shuffle):
            if(round >= 3):
                step1_finish = 1
                # good_group and bad_group start
        
        print("Acc_Avg (total global): {}".format(acc_avg))
        std_avg = sum(acc_per_label_std)/len(acc_per_label_std)
        print("Acc_Std (total global): {}".format(std_avg))
        print("")            

        
        if(args.shuffle):

            if(step1_finish==1):
                for client, acc in acc_list[:int(args.num_clients*0.4)]:
                    shuffle_in_good.append(client)
                    
                for client, acc in acc_list[int(args.num_clients*0.4):args.num_clients-int(args.num_clients*0.3)]:
                    shuffle_in_normal.append(client)
                
                for client, acc in acc_list[args.num_clients-int(args.num_clients*0.3):]:        
                    shuffle_in_bad.append(client)
                    
            else:
                for client in range(args.num_clients): 
                    shuffle_in_normal.append(client)
                            

        good = shuffle_in_good  
        normal = shuffle_in_normal 
        bad = shuffle_in_bad 

        acc_good_list = []
        if(len(good)>0):
            for g in good:
                acc_good_list.append(acc_per_label_avg[g])
            np_acc = np.array(acc_good_list)
            good_acc = np.mean(np_acc)
            print('good_group_acc : ',good_acc)

        if(first_save==0 and len(good)>=args.num_clients*0.3):
                print('first save models')
                rec_good = copy.deepcopy(shuffle_in_good)
                rec_normal = copy.deepcopy(shuffle_in_normal)
                rec_bad = copy.deepcopy(shuffle_in_bad)
                rec_round = round
                for i in shuffle_in_good:
                        path = args.model_path + '(' + str(i) + ')' + '.pt1'
                        torch.save(net_glob_backup[i].state_dict(), path)
                        rec_locals[i] = local_users[i]
                        best_acc = good_acc
                        first_save = 1
        elif(first_save==1 and len(good)>=args.num_clients*0.3 and best_acc<good_acc):
                print('models updates')
                print('before: ',best_acc, ' now:',good_acc)
                rec_good = copy.deepcopy(shuffle_in_good)
                rec_normal = copy.deepcopy(shuffle_in_normal)
                rec_bad = copy.deepcopy(shuffle_in_bad)
                rec_round = round
                for i in shuffle_in_good:
                        path = args.model_path + '(' + str(i) + ')' + '.pt1'
                        torch.save(net_glob_backup[i].state_dict(), path)
                        rec_locals[i] = local_users[i]
                        best_acc = good_acc    
                for i in shuffle_in_normal:
                        rec_locals[i] = local_users[i]
                for i in rec_normal:
                        print('normal: ',rec_locals[i])
                        print("")
        if(args.shuffle and round+1<args.epochs):
            print('shuffle_clients (in good):', shuffle_in_good)
            shuffle_user_in_good = []
            if(len(shuffle_in_good)>1):
                print('Shuffle in good !')
                for i in shuffle_in_good:
                    shuffle_user_in_good.extend(local_users[i])
                for i in shuffle_in_good:
                    local_users[i] = set(np.random.choice(shuffle_user_in_good, args.num_users, replace=False))
                    shuffle_user_in_good = list(set(shuffle_user_in_good) - local_users[i])
        print("")
        
        

        if(args.shuffle and round+1<args.epochs):
            print('shuffle_clients (in normal):', shuffle_in_normal)
            shuffle_user_in_normal = []
            if(len(shuffle_in_normal)>1):
                print('Shuffle in normal!')
                for i in shuffle_in_normal:
                    shuffle_user_in_normal.extend(local_users[i])
                for i in shuffle_in_normal:
                    local_users[i] = set(np.random.choice(shuffle_user_in_normal, args.num_users, replace=False))
                    shuffle_user_in_normal = list(set(shuffle_user_in_normal) - local_users[i])
        print("")

        if(args.shuffle and round+1<args.epochs):
            print('shuffle_clients (in bad):', shuffle_in_bad) 
            shuffle_user_in_bad = []
            if(len(shuffle_in_bad)>1):
                print('Shuffle in bad !')
                for i in shuffle_in_bad:
                    shuffle_user_in_bad.extend(local_users[i])
                for i in shuffle_in_bad:
                    local_users[i] = set(np.random.choice(shuffle_user_in_bad, args.num_users, replace=False))
                    shuffle_user_in_bad = list(set(shuffle_user_in_bad) - local_users[i])
        print("")
            
        end_time = time.time()
        print("max_local_ep_time: ",max_local_ep_time)
        round_time = max_local_ep_time + global_test_time + end_time - start_time
        print("round_time: ",round_time)
        print("")

        total_time += round_time
        
    print('Round: ', rec_round)
    print('save {} models'.format(len(rec_good)))
    for i in rec_good:
        print('locals: ',rec_locals[i])
        print("")
    
    print('clients (in good):', rec_good)
    print('clients (in normal):', rec_normal)
    print('clients (in bad):', rec_bad)
    
    print("total_time: ",total_time)                                
    print("=== End ===")
