import argparse
from statistics import mode
import torch 
import numpy as np
import time 
import matplotlib.pyplot as plt

# from net import LinearNet
from constrastive import ContrastiveLoss
from torch.autograd import Variable
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
import os 
import pickle 
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..models.embedding_models import LinearNet
import numpy as np
from net import LinearNet, Net 

class HiddenStatesAPMsDataset(Dataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.hiddenstates_dir = self.data_dir + '/HiddenStatesDB'
        self.apms_dir = self.data_dir + '/APMsDB'

    def __len__(self):
        return len([name for name in os.listdir(self.apms_dir)])

    def __getitem__(self, index): 
        with open(self.hiddenstates_dir + "/" + str(index) + ".pickle", 'rb') as f: 
            hiddenstates = pickle.load(f)
        with open(self.apms_dir + "/" + str(index) + ".pickle", 'rb') as f: 
            apms = pickle.load(f)
        
        return {"hiddenstates":hiddenstates, "apms": apms}

 
def create_pairs(inputs):
    x0_data = []
    x1_data = []
    labels = []
    data = list(combinations(inputs['hiddenstates'], 2))
    sample = list(combinations(inputs['apms'], 2)) 
    for (x0, x1), (apm0, apm1) in zip(data, sample):

        # apm00 = apm0.squeeze(0).numpy()[0] # head 0
        # apm11 = apm1.squeeze(0).numpy()[0]
        # seq_len = apm00.shape[0]
        # label = np.sum(np.abs(apm00-apm11)) / seq_len / 2

        apm0 = apm0.numpy() # 24 heads 
        apm1 = apm1.numpy()
        batch_size, num_head, seq_len = apm0.shape[0], apm0.shape[1], apm0.shape[2]
        diff = np.abs(apm0-apm1)
        label = np.sum(diff, axis=tuple(range(1, diff.ndim))) / seq_len  / num_head / 2 
          

    
        x0_data.append(x0)
        x1_data.append(x1)
        labels.append(label)
    x0_data = torch.cat(x0_data, dim=0)
    x1_data = torch.cat(x1_data, dim=0)
    labels = np.concatenate(labels, axis=0)
    labels = torch.from_numpy(np.array(labels, dtype=np.float32))
    return x0_data, x1_data, labels

def main():
    torch.manual_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dataset_dir', action='store_true', default="/home/sdh/Llama-3.2-3B-Instruct_DBs/chemistry_5shots_18layers",
                        help='dataset_dir')
    parser.add_argument('--shots', type=int, default=0,
                        help='prompt shots')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.cuda = False
    args.dataset_dir = "/home/sdh/MoE_Embedding/MoE-Embedding/database/Llama-3.2-3B-Instruct"
    args.shots = 0

    print("Args: %s" % args)

    # model = Net()
    model = LinearNet(args.shots)
    # model = SingleNet()
    if args.cuda:
        model.cuda() 

    learning_rate = 0.1
    momentum = 0.9

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    datasets = HiddenStatesAPMsDataset(args.dataset_dir)
    loss_fn = ContrastiveLoss()

    # if args.cuda:
    #     kwargs = {'num_workers':1, 'pin_memory':True} #当 pin_memory=True 时，DataLoader 会将数据加载到内存中，并将数据固定（pinned）到 GPU 可访问的内存中。
    # else:
    kwargs = {}
    
    train_loader = DataLoader(datasets, batch_size=args.batchsize, shuffle=True)
    def train(epoch):
        train_loss = []
        model.train() 
        start = time.time() 
        for batch_idx, inputs in enumerate(train_loader):
            x0, x1, labels = create_pairs(inputs)
            if args.cuda:
                x0, x1, labels = x0.cuda(), x1.cuda(), labels.cuda()
            x0, x1, labels = Variable(x0), Variable(x1), Variable(labels)
            output1, output2 = model(x0, x1)
            loss = loss_fn(output1, output2, labels)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print("Loss: {}, batch:{}, Epoch:{}, Time/batch = {}".format(loss.item(), batch_idx, epoch, time.time()-start))
                torch.save(model.state_dict(), f'{args.dataset_dir}/Embedding_models/mlp_model-epoch{epoch}.pth')
                print(f"Save embedding model to {args.dataset_dir}/Embedding_models/mlp_model-epoch{epoch}.pth")
  
        torch.save(model.state_dict(), f'{args.dataset_dir}/Embedding_models/mlp_model-epoch{epoch}.pth')
        print(f"Save embedding model to {args.dataset_dir}/Embedding_models/mlp_model-epoch{epoch}.pth")
        return train_loss
    
    train_loss = []
    for epoch in range(1, args.epoch + 1):
        train_loss.extend(train(epoch))
    
    plt.gca().cla()
    plt.plot(train_loss, label="train loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss with MLP")
    plt.legend()
    plt.draw()
    plt.savefig('MLP_train_loss.png')
    plt.gca().clear()

if __name__ == "__main__" :
    start = time.time()
    main()
    print("Training Time ",time.time() - start)

