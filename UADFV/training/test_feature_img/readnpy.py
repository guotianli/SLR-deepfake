import numpy as np

# 假设你要读取的npy文件路径，这里需要根据实际情况修改  capsule core  effnb4  f3net srm xception
npy_file_path = 'batch_0_xception.npy'

# 加载npy文件
data = np.load(npy_file_path, allow_pickle=True).item()

# 显示predictions的内容
print("Predictions:")
print(data['predictions'])

# 显示labels的内容
print("\nLabels:")
print(data['labels'])

# 显示features的形状
print("\nFeatures Shape:")
print(data['features'].shape)

# 显示image_paths的内容
print("\nImage Paths:")
print(data['image_paths'])