import torch
from model import RouteNet
from Runner import Runner
from simple_network_dataset import NetworkDataset,get_dataloader
from weight_init import weight_init
import os


train_path = os.path.join(os.getcwd(),"path to train")
test_path = os.path.join(os.getcwd(),"path to val")

train_set = NetworkDataset(train_path)
test_set = NetworkDataset(test_path)

model = RouteNet()
model.apply(weight_init)

opts_dict = {
             'lr': .001,
             'batch_size': 1,
             'epochs': 15
            }

max_vals = torch.load(os.path.join(os.getcwd(),"max_values.pt"))

runner = Runner(model,train_set,test_set,opts_dict,max_vals,log_name = "run_2")
runner.train()