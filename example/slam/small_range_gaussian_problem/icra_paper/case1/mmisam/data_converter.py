import numpy as np
for i in range(6):
    res = np.load(f'batch{i+1}.npz')
    np.savetxt(f'batch{i+1}',X=res.T)
res = np.load('timing.npz')
np.savetxt('timing',X=res)