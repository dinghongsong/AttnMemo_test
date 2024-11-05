import torch
print(f'Pytorch version : {torch.__version__}')
print(*torch.__config__.show().split("\n"), sep="\n")

from torchvision import models
from torch.utils import mkldnn as mkldnn_utils
import time

def forward(net, use_mkldnn=False, iteration=1, batch_size=10, weight_cache = False):
  net.eval()
  batch = torch.rand(batch_size, 3,224,224)
  if use_mkldnn:
    net = mkldnn_utils.to_mkldnn(net)
    batch = batch.to_mkldnn()
    if weight_cache:
        # using weight cache which will reduce weight reorder
        fname = 'test.script.pt'
        traced = torch.jit.trace(net, batch, check_trace=False)
        script = traced.save(fname)
        net = torch.jit.load(fname)

  start_time = time.time()
  for i in range(iteration):
      with torch.no_grad():
          net(batch)
  return time.time() - start_time

net = models.resnet18(False)
iter_cnt = 100
batch_size = 1
no_mkldnn   = forward(net, False, iter_cnt, batch_size)
with_mkldnn = forward(net, True,  iter_cnt, batch_size)
with_mkldnn_cache = forward(net, True,  iter_cnt, batch_size, True)

print(f"time-normal: {no_mkldnn:.4f}s")
print(f"time-mkldnn: {with_mkldnn:.4f}s")
print(f"time-mkldnn with weight cache: {with_mkldnn_cache:.4f}s")
print(f"mkldnn is {no_mkldnn/with_mkldnn:.2f}x faster!")
print(f"mkldnn using weight cache is {no_mkldnn/with_mkldnn_cache:.2f}x faster!")