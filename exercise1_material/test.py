import numpy as np

[1,2,3,4]

# 4 Samples, 20 features each
input_tensor = np.random.uniform(0,1,(4,20))

ones = np.ones(shape=(input_tensor.shape[0],1))

print(ones.shape)

fuckthisshit = np.hstack((input_tensor, ones))

print(fuckthisshit.shape)