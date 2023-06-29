import numpy as np

# 指定维度
shape = (3, 4)  # 3行4列的随机数

# 生成服从伯努利分布的随机数
random_array = np.random.binomial(n=1, p=0.5, size=shape)

print(random_array)