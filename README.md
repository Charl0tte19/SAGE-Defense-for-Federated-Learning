# SAGE-Defense-for-Federated-Learning

Implementation of Shuffling and Regrouping Based Defense for Federated Learning

## Prerequisites

Our experiments were run on Ubuntu server with GeForce RTX 3090. We use Python 3.8 and the required packages are listed as follows.

- pytorch 1.8.0
- torchvision 0.9.0
- torchaudio 0.8.0
- cudatoolkit 11.1
- scikit-learn 0.24.1
- scipy 1.5.4
- matplotlib 3.3.4
- numpy 1.19.5

You can also build the environment and install the requirements as follows.

```
conda create -n SAGE python=3.8
source activate SAGE
pip install -r requirements.txt 
```

## Reproduce the results

### - Run a single case

#### For example, run the case (Fashion-MNIST, non-IID degree K = 0.4, attack ratio = 0.2)

run the following code in ./code/

1. Train the model by SAGE.

```
python -u main_shuffle_fmnist.py --gpu 0 --seed 32 --dataset="fmnist" --epoch 20 --noniid 0.4 --attack_mode="poison" --attack_ratio 0.2 --test_label_acc --target_random --shuffle --model_path="./fmnist/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.pt" 2>&1 | tee ./fmnist/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.txt
```

2. Create the directories ./code/fmnist/noniid_0.4/ and ./code/fmnist/noniid_0.8/.

```
bash ./fmnist/make_dir.sh
```

3. Continue to train the model by FL after SAGE. Assume that the best master model is No.6 master model.

```
python -u after_preprocess_FL_fmnist.py --dataset=fmnist --noniid 0.4 --seed 32 --epoch 80 --attack_mode=poison --attack_ratio 0.2 --test_label_acc --target_random --model_path=./fmnist/noniid_0.4/ratio_0.2/final.pt --pretrained_model=./fmnist/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.pt(6).pt1 --local_file=./fmnist/noniid_0.4/ratio_0.2/local.txt --attacker_file=./fmnist/noniid_0.4/ratio_0.2/attacker.txt 2>&1 | tee ./fmnist/noniid_0.4/ratio_0.2/log.txt
```

4. Train the model by typical FL.

```
python -u main_origin_fmnist.py --gpu 0 --seed 32 --epoch 100 --noniid 0.4 --attack_mode="poison" --attack_ratio 0.2 --test_label_acc --target_random --dataset="fmnist" --model_path="./fmnist/origin/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.pt" 2>&1 | tee ./fmnist/origin/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.txt
```

___


#### For another example, run the case (MNIST, non-IID degree K = 0.8, attack ratio = 0.02)

run the following code in ./code/

1. Train the model by SAGE.

```
python -u main_shuffle_mnist.py --gpu 0 --seed 32 --dataset="mnist" --scale --epoch 20 --noniid 0.8 --attack_mode="poison" --attack_ratio 0.02 --test_label_acc --target_random --shuffle --model_path="./mnist/noniid_0.8/ratio_0.02/poison_${ratio4}0.02_Scale_0.pt" 2>&1 | tee ./mnist/noniid_0.8/ratio_0.02/poison_0.02_Scale_0.txt
```

2. Create the directories ./code/mnist/noniid_0.4/ and ./code/mnist/noniid_0.8/.

```
bash ./mnist/make_dir.sh
```

3. Continue to train the model by FL after SAGE. Assume that the best master model is No.6 master model.

```
python -u after_preprocess_FL_mnist.py --seed 12 --epoch 80 --noniid 0.8 --attack_ratio 0.02 --test_label_acc --target_random --scale --model_path=./mnist/seed_12/noniid_0.8/ratio_0.02/final.pt --pretrained_model=./mnist/seed_12/noniid_0.8/ratio_0.02/poison_0.02_Scale_0.pt(4).pt1 --local_file=./mnist/seed_12/noniid_0.8/ratio_0.02/local.txt --attacker_file=./mnist/seed_12/noniid_0.8/ratio_0.02/attacker.txt 2>&1 | tee ./mnist/noniid_0.8/ratio_0.02/log.txt
```

4. Train the model by typical FL.

```
python -u main_origin_mnist.py --gpu 0 --seed 32 --scale --epoch 100 --noniid 0.8 --attack_mode="poison" --attack_ratio 0.02 --test_label_acc --target_random --dataset="mnist" --model_path="./mnist/origin/noniid_0.8/ratio_0.02/poison_0.02_Scale_0.pt" 2>&1 | tee ./mnist/origin/noniid_0.8/ratio_0.02/poison_0.02_Scale_0.txt
```
