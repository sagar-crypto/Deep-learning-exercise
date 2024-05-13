import numpy as np

# Just a testfile for myself to get some things debugged
[1,2,3,4]

# 4 Samples, 20 features each
input_tensor = np.random.uniform(0,1,(4,20))

#ones = np.ones(shape=(input_tensor.shape[0],1))
ones = np.ones((1, input_tensor.shape[0])).T
print(ones.shape)

fuckthisshit = np.hstack((input_tensor, ones))

anothershit = np.concatenate((input_tensor, ones), axis=1)

print(fuckthisshit.shape)