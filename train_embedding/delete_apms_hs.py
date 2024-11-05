
import os
import pickle
data_dir = "/home/sdh/MoE_Embedding/MoE-Embedding/database/Llama-3.2-3B-Instruct"
files_num = len(os.listdir(f"{data_dir}/APMsDB"))
for i in range(files_num):  
    with open(f"{data_dir}/HiddenStatesDB" + "/" + str(i) + ".pickle", 'rb') as f: 
            hiddenstates = pickle.load(f)
            num = hiddenstates.shape[0]
    if num == 58:
        os.remove(f"{data_dir}/HiddenStatesDB" + "/" + str(i) + ".pickle")
        os.remove(f"{data_dir}/APMsDB" + "/" + str(i) + ".pickle")
        