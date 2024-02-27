import numpy as np

output=np.fromfile(r'gpu_out/Result_0/output.raw', dtype='float32')
output = np.squeeze(output)
dst = np.where(output==np.max(output))
print('output:', output)
print('dst:', dst[0][0])

