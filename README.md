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

1. Create the directories ./fmnist/noniid_0.4/ and ./fmnist/noniid_0.8/.

```
bash ./fmnist/make_dir.sh
```

2. Train the model by SAGE.

```
python -u main_shuffle_fmnist.py --gpu 0 --seed 32 --dataset="fmnist" --epoch 20 --noniid 0.4 --attack_mode="poison" --attack_ratio 0.2 --test_label_acc --target_random --shuffle --model_path="./fmnist/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.pt" 2>&1 | tee ./fmnist/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.txt
```

3. Continue to train the model by FL after SAGE. Assume that the best master model is No.6 master model.

```
python -u after_preprocess_FL_fmnist.py --dataset=fmnist --noniid 0.4 --seed 32 --epoch 80 --attack_mode=poison --attack_ratio 0.2 --test_label_acc --target_random --model_path=./fmnist/noniid_0.4/ratio_0.2/final.pt --pretrained_model=./fmnist/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.pt(6).pt1 --local_file=./fmnist/noniid_0.4/ratio_0.2/local.txt --attacker_file=./fmnist/noniid_0.4/ratio_0.2/attacker.txt 2>&1 | tee ./fmnist/noniid_0.4/ratio_0.2/log.txt
```

4. Train the model by typical FL.

```
mkdir ./fmnist/origin/

cp ./fmnist/make_dir.sh ./fmnist/origin

bash ./fmnist/origin/make_dir.sh

python -u main_origin_fmnist.py --gpu 0 --seed 32 --epoch 100 --noniid 0.4 --attack_mode="poison" --attack_ratio 0.2 --test_label_acc --target_random --dataset="fmnist" --model_path="./fmnist/origin/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.pt" 2>&1 | tee ./fmnist/origin/noniid_0.4/ratio_0.2/poison_0.2_notScale_0.txt
```

5. Test the trained global model.

```
python -u test_trained_models_mnist.py --gpu 0 --seed 32 --target_random --test_label_acc --final_model="./mnist/noniid_0.8/ratio_0.02/final.pt" 2>&1 | tee ./mnist/seed_32/noniid_0.4/ratio_0.02/testing.txt
```
___


#### For another example, run the case (MNIST, non-IID degree K = 0.8, attack ratio = 0.02)

run the following code in ./code/

1. Create the directories ./mnist/noniid_0.4/ and ./mnist/noniid_0.8/.

```
bash ./mnist/make_dir.sh
```

2. Train the model by SAGE.

```
python -u main_shuffle_mnist.py --gpu 0 --seed 32 --dataset="mnist" --scale --epoch 20 --noniid 0.8 --attack_mode="poison" --attack_ratio 0.02 --test_label_acc --target_random --shuffle --model_path="./mnist/noniid_0.8/ratio_0.02/poison_${ratio4}0.02_Scale_0.pt" 2>&1 | tee ./mnist/noniid_0.8/ratio_0.02/poison_0.02_Scale_0.txt
```

3. Continue to train the model by FL after SAGE. Assume that the best master model is No.6 master model.

```
python -u after_preprocess_FL_mnist.py --seed 12 --epoch 80 --noniid 0.8 --attack_ratio 0.02 --test_label_acc --target_random --scale --model_path=./mnist/seed_12/noniid_0.8/ratio_0.02/final.pt --pretrained_model=./mnist/seed_12/noniid_0.8/ratio_0.02/poison_0.02_Scale_0.pt(4).pt1 --local_file=./mnist/seed_12/noniid_0.8/ratio_0.02/local.txt --attacker_file=./mnist/seed_12/noniid_0.8/ratio_0.02/attacker.txt 2>&1 | tee ./mnist/noniid_0.8/ratio_0.02/log.txt
```

4. Train the model by typical FL.

```
mkdir ./mnist/origin/

cp ./mnist/make_dir.sh ./mnist/origin

bash ./mnist/origin/make_dir.sh

python -u main_origin_mnist.py --gpu 0 --seed 32 --scale --epoch 100 --noniid 0.8 --attack_mode="poison" --attack_ratio 0.02 --test_label_acc --target_random --dataset="mnist" --model_path="./mnist/origin/noniid_0.8/ratio_0.02/poison_0.02_Scale_0.pt" 2>&1 | tee ./mnist/origin/noniid_0.8/ratio_0.02/poison_0.02_Scale_0.txt
```

5. Test the trained global model.

```
python -u test_trained_models_mnist.py --gpu 0 --seed 32 --target_random --test_label_acc --final_model="./mnist/noniid_0.8/ratio_0.02/final.pt" 2>&1 | tee ./mnist/seed_32/noniid_0.4/ratio_0.02/testing.txt
```
___

### - Run all cases

#### For example, run all cases of Fashion-MNIST.

run the following code in ./code/

1. Move all sh scripts from ./sh_scripts_fmnist/ to ./

2. Create the directories ./fmnist/noniid_0.4/ and ./fmnist/noniid_0.8/.

```
bash ./fmnist/make_dir.sh
```

3. Train the model by SAGE.

```
bash ./shuffle1_fmnist.sh
bash ./shuffle2_fmnist.sh
bash ./shuffle3_fmnist.sh
bash ./shuffle4_fmnist.sh
```

4. Create the directories for the chosen seed. Assume the seed is 32.

```
mkdir ./fmnist/seed_32

cp ./fmnist/file.sh ./fmnist/seed_32/

mkdir ./fmnist/seed_32/command/

bash ./fmnist/seed_32/file.sh
```

4. Continue to train the model by FL after SAGE. Assume the seed is 32.

```
bash ./after1_fmnist.sh
bash ./after2_fmnist.sh
bash ./after3_fmnist.sh
bash ./after4_fmnist.sh
```

5. Train the model by typical FL.

```
mkdir ./fmnist/origin/

cp ./fmnist/make_dir.sh ./fmnist/origin

bash ./fmnist/origin/make_dir.sh

bash ./origin1_fmnist.sh
bash ./origin2_fmnist.sh
bash ./origin3_fmnist.sh
bash ./origin4_fmnist.sh
```

Please check the path before running each sh script.

## Result Charts

We show the complete version of the result charts in [result/](https://github.com/Charl0tte19/SAGE-Defense-for-Federated-Learning/tree/main/result).

#### - Example ( MNIST, non-IID degree K = 0.4, case LD )

 Attack ratio  | Number of attackers assigned to each subset in each group | Average number of attackers assigned to each subset in each group | Validation accuracy
 ------------- |-------------| ----------- | --------------
 0.3      | ![alt text](https://raw.githubusercontent.com/Charl0tte19/SAGE-Defense-for-Federated-Learning/main/result/chart/amount/noniid_0.4_ratio_0.3_mnist_regroup.png) | ![alt text](https://raw.githubusercontent.com/Charl0tte19/SAGE-Defense-for-Federated-Learning/main/result/chart/amount/noniid_0.4_ratio_0.3_mnist_attacker_in_group.png) | ![alt text](https://github.com/Charl0tte19/SAGE-Defense-for-Federated-Learning/blob/main/result/chart/accuracy/noniid_0.4_ratio_0.3_mnist_acc.png)

## Comments of each file

```
code/                      
    fmnist/                
        file.sh                              # code of creating the orders to run after*.sh for all cases.
        make_dir.sh                          # code for creating directories noniid_0.4/ and noniid_0.8/
        temp.py                              # code that supporting file.sh
    mnist/                 
        file.sh                              # code of creating the orders to run after*.sh for all cases.
        make_dir.sh                          # code for creating directories noniid_0.4/ and noniid_0.8/
        temp.py                              # code that supporting file.sh
    models/                
        Fed.py                               # code of FedAvg
        Nets.py                              # code of CNN models
        Update.py                            # code of poisoning, training and testing the models for mnist
        Update2.py                           # code of poisoning, training and testing the models for fmnist
        test.py                              # code of validating the master models for mnist
        test2.py                             # code of validating the master models for fmnist 
    sh_scripts_fmnist/
        after*.sh                            # script for running after_preprocess_FL_fmnist.py for all cases
        origin*.sh                           # script for running main_origin_fmnist.py for all cases
        shuffle*.sh                          # script for running main_shuffle_fmnist.py for all cases
    sh_scripts_mnist/
        after*.sh                            # script for running after_preprocess_FL_mnist.py for all cases
        origin*.sh                           # script for running main_origin_mnist.py for all cases
        shuffle*.sh                          # script for running main_shuffle_mnist.py for all cases
    utils/
        sampling.py                          # code of sampling the dataset
    after_preprocess_FL_fmnist.py            # code of training the global model after SAGE for fmnist.
    after_preprocess_FL_mnist.py             # code of training the global model after SAGE for mnist.
    main_origin_fmnist.py                    # code of training the master models by typical FL for fmnist.
    main_origin_mnist.py                     # code of training the master models by typical FL for mnist.
    main_shuffle_fmnist.py                   # code of training the master models by SAGE for fmnist.
    main_shuffle_mnist.py                    # code of training the master models by SAGE for mnist.
    test_trained_models_fmnist.py            # code of testing the trained global model for fmnist.
    test_trained_models_mnist.py             # code of testing the trained global model for mnist.
result/
    chart/                                   # *.png of result charts
    fmnist/                                  # the master models and training logs for fmnist (seed_12)
    mnist/                                   # the master models and training logs for mnist (seed_12)
    acc_between_SAGE_and_FL.ipynb            # jupyter notebook for drawing accuracy
    attacker_in_each_group.ipynb             # jupyter notebook for drawing amount of attackers assigned to each subset in each group
    avg_attacker_in_each_group.ipynb         # jupyter notebook for drawing average amount of attackers assigned to each subset in each group
```
