# DMIR

The PyTorch implementation of paper Debiased Model-based Interactive Recommendation. We conducted experiments on three public datasets, Ciao, Epinions and Yelp. The following takes Ciao as an example.

│

├─common

│ Datasets.py

│ memory.py

│ model.py

│ utils.py

│

├─datasets

│ ├─Ciao

│ │ data\_pre.py

│ │ rating\_with\_timestamp.mat

│ │ trust.mat

│ │

│ ├─Epin

│ │ data\_pre.py

│ │ rating\_with\_timestamp.mat

│ │ trust.mat

│ │

│ └─Yelp

│ data\_pre.py

│ yelp\_academic\_dataset\_review.json

│ yelp\_academic\_dataset\_user.json

│

├─dmir

│ agent.py

│ main.py

│

├─envs

│ CA\_model.py

│ counter\_env.py

│ ground\_truth.py

│ pre\_train.py

│ utils.py

│

└─GroundTruth

ground\_truth.py

##Environment
We trained our models on one server with five Intel(R) Xeon(R) Gold 5218 CPU with 50GB CPU memory and a single V100 GPU with 32GB GPU memory. In our experiments, we use python 3.6, torch 1.9.1 with CUDA version 11.4.
## Run

### Step 1.Data Preprocess

```
Cd datasets/Ciao
Python data_pre.py
```

### Step 2.Get Ground Truth

```
cd GroundTruth 
python ground_truth.py --dataset Ciao
```

### Step 3. Pretrain Debiased Causal World Model

```
cd envs 
python pre_train.py --dataset Ciao
```

### Step 4. Train and Test

```
cd dmir
python main.py --dataset Ciao
```

Due to file size limitations, the Yelp dataset is not included here. Please go to ([https://www.kaggle.com/yelp-dataset/yelp-dataset](https://www.kaggle.com/yelp-dataset/yelp-dataset)) to download Yelp to the datasets/Yelp path.



