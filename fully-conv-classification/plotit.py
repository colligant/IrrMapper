import numpy as np
import matplotlib.pyplot as plt


# grep val_f1: metrics | awk '{print $8, $11, $14, $17, $20, $23}'  > val_acc_and_f1.txl

arr = np.genfromtxt('val_acc_and_f1.txl')
print(arr.shape)

fig, ax = plt.subplots(nrows=3,figsize=(13, 10))

ax[0].plot(arr[:, 0], label='loss')
ax[0].plot(arr[:, 3], label='val_loss')

ax[1].plot(arr[:, 1], label='acc')
ax[1].plot(arr[:, 4], label='val_acc')

ax[2].plot(arr[:, 2], label='f1')
ax[2].plot(arr[:, 5], label='val_f1')

ax[1].legend()
ax[0].legend()
plt.show()
