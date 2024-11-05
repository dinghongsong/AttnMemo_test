import redis
import numpy as np

# 生成一个随机整数，范围为 [0, 10)
rand_int = np.random.rand(3, 3)
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)  
r.set('mykey', rand_int)
value = r.get('mykey')  
print(value)  # 输出: myvalue

print("ok")