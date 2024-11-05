# % imports 
import faiss
import numpy as np 
import pickle
from net import LinearNet, Net 
import torch
from tqdm import tqdm




# % Define the embedding
class Emb():
    def __init__(self, model_dir, shots) -> None:
        self.emb = LinearNet(shots=shots)  
        self.emb.load_state_dict(torch.load(model_dir))
        self.emb.eval()
    def embed(self, inputs):
        """
        return a numpy obj
        """
        return self.emb.forward_once(torch.from_numpy(inputs)).detach().numpy()
# % VecDB
class VecDB():
    def __init__(self, d=128, nlist=128, m=8, bit=8) -> None:
        ''''
        d dimention number
        nlist n prob
        m 
        bit PQ 
        IndexIVFPQ for less memory
        '''
        quantizer = faiss.IndexFlatL2(d)
        # self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bit)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
        # self.index = faiss.IndexFlatL2(d) 
    def train_index(self, embeddings):
        assert not self.index.is_trained
        self.index.train(embeddings)
        assert self.index.is_trained

    def add(self, embedding):
        self.index.add(embedding)

    def search(self, embedding, k=1, nprob = 128):
        self.index.nprob = nprob
        D, I = self.index.search(embedding, k)#I为每个待检索query最相似TopK的索引list，D为其对应的距离
        return D, I
    
    def load(self, path):
        self.index = faiss.read_index(path)
    
    def save(self, save_dir, epoch):
        faiss.write_index(self.index, f"{save_dir}/mlp_epoch{epoch}_vectors.faiss")
        print(f"Save vecdb to {save_dir}/mlp_epoch{epoch}_vectors.faiss success!")
        

import torch
import os
# data_dir = "/home/sdh/Llama-3.2-3B-Instruct_DBs/chemistry_5shots_18layers"
# epoch = 1


data_dir = "/home/sdh/MoE_Embedding/MoE-Embedding/database/Llama-3.2-3B-Instruct"
epoch = 6

print(f"using embedding model: {data_dir}/Embedding_models/mlp_model-epoch{epoch}.pth")
emb = Emb(f"{data_dir}/Embedding_models/mlp_model-epoch{epoch}.pth", shots=0)
files_num = len(os.listdir(f"{data_dir}/HiddenStatesDB"))

for i in tqdm(range(files_num)):
    with open(f"{data_dir}/HiddenStatesDB/" + str(i) + ".pickle", 'rb') as f:
        hidden = pickle.load(f)
    if i == 0:
        layer_tensor =  emb.embed(hidden)
    else:
        tmp = emb.embed(hidden)
        layer_tensor = np.vstack((layer_tensor, tmp))

vecdb = VecDB()
vecdb.train_index(layer_tensor)
vecdb.add(layer_tensor)
vecdb.save(save_dir=f"{data_dir}/VectorDB", epoch=epoch)
print("Save success!")



