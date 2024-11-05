import time
import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils import mkldnn as mkldnn_utils

class LinearNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(24576, 128)
        self.fc2 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d(128)
    
    def forward_once(self, x):
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

class SingleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24576, 128)
        self.fc2 = nn.Linear(128, 128)
#        self.bn1 = nn.BatchNorm2d(1)
#        self.bn2 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv2d(1,1,3)

    def forward_once(self, x):
        # x = torch.unsqueeze(x,1)
        x = self.conv1(x)
#        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
#        x = self.bn2(x)
        x = self.fc2(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,1,3)
        self.conv2 = nn.Conv2d(1,1,3)
        self.fc1 = nn.Linear(23684, 128)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)

    def forward_once(self, x):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        end.record()
        torch.cuda.synchronize()
        print("Embedding Time",start.elapsed_time(end), " ms")
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size =768
        num_attention_heads = 12
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(768 / 12)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.self_attention_profiling = 1

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        relevance_mask = None,
    ):
        attn_start = time.time()
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        ################################################################### X
        transpose_start = time.time()

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        transpose_end = time.time()
        print("Transpose Linear Time is ", transpose_end - transpose_start)

        ################################################################ X
        qk_matmul_start = time.time()

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # shape torch.Size([1, 12,384,384]) 
                                                                                  # dtype torch.float32

        qk_matmul_end = time.time()
        print("QK Matmul Time is ", qk_matmul_end - qk_matmul_start)
        ############################################################### X
        Norm_head_start = time.time()

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        Norm_head_end = time.time()
        print("Norm Attn Head Time", Norm_head_end - Norm_head_start)

        #############################################################  X
        Attention_masking_start = time.time()
        #! Here we do the relevance masking
        if relevance_mask is not None:
            attention_scores = relevance_mask + attention_scores

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        Attention_masking_end = time.time()
        print("Attention_maskingTime =", Attention_masking_end - Attention_masking_start)
        ##################################################  X
        # Normalize the attention scores to probabilities.
        Softmax_start = time.time()

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        Softmax_end = time.time()
        print("Softmax time = ", Softmax_end - Softmax_start)
        print("Replaceable Time = ", Softmax_end - attn_start)
        ##################################################### X
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        Dropout_start = time.time()
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        Dropout_end = time.time()
        print("Dropout time = ", Dropout_end - Dropout_start)

        ################################ X
        Context_start = time.time()
        context_layer = torch.matmul(attention_probs, value_layer)
        Context_end = time.time()
        print("Context_layer = ", Context_end - Context_start)

        ########################### X
        Reshape_start = time.time()
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        Reshape_end = time.time()
        print("Reshape time:" , Reshape_end - Reshape_start)
        ############################
        Bert_SelfAttention_end = time.time()
        print("Bert_Self_attention", Bert_SelfAttention_end - attn_start)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

if __name__ == "__main__":
    inp0 = torch.rand((64, 1, 128, 768))
    inp1 = torch.rand((64, 128, 768))
    # inp = inp.cuda()

    net = LinearNet()
    net.eval()
    # # net = net.cuda()
    start = time.time()
    for i in range(100):
        net.forward_once(inp0)
    print((time.time() - start)/100)

    # print("Time on MKLDNN")
    # net = SingleNet()
    # net.eval()
    # input1 = inp0.to_mkldnn()
    # net = mkldnn_utils.to_mkldnn(net)
    # start = time.time()
    # for i in range(100):
    #     with torch.no_grad():
    #         net.forward_once(input1)
    # print((time.time() - start)/100)

    net = Net()
    net.eval()
    # net = net.cuda()
    net.forward_once(inp1)

    attention = BertSelfAttention()
    attention.eval()
    # attention = attention.cuda()
    attention(inp1)
 
