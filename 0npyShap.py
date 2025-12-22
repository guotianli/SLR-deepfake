import os
import glob
import numpy as np


def inspect_npy_file(file_path):
    """检查单个npy文件的结构和内容"""
    print(f"\n正在检查文件: {file_path}")
    try:
        # 加载npy文件
        data = np.load(file_path, allow_pickle=True).item()

        # 打印基本信息
        print(f"数据类型: {type(data)}")
        print(f"包含的键: {data.keys()}")

        # 检查每个键的信息
        for key in data.keys():
            value = data[key]
            print(f"\n键: {key}")
            print(f"  类型: {type(value)}")

            if isinstance(value, np.ndarray):
                print(f"  形状: {value.shape}")
                print(f"  数据类型: {value.dtype}")
                print(f"  示例值: {value[:5] if value.size > 0 else '空数组'}")
            elif isinstance(value, (list, tuple)):
                print(f"  长度: {len(value)}")
                if len(value) > 0 and isinstance(value[0], np.ndarray):
                    print(f"  元素类型: numpy数组, 形状: {value[0].shape}")
                    print(f"  示例值: {value[0][:5] if value[0].size > 0 else '空数组'}")
            else:
                print(f"  值: {value[:50] if isinstance(value, str) else value}")  # 限制输出长度

    except Exception as e:
        print(f"检查文件时出错: {e}")


def inspect_npy_folder(folder_path):
    """检查文件夹中所有npy文件的结构"""
    print(f"正在检查文件夹: {folder_path}")
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))

    if not npy_files:
        print(f"警告: 文件夹中没有找到npy文件")
        return

    print(f"找到 {len(npy_files)} 个npy文件")

    for i, file_path in enumerate(npy_files):
        print(f"\n---- 文件 {i+1}/{len(npy_files)} ----")
        inspect_npy_file(file_path)


def main():
    """主函数"""
    # 设置要检查的文件夹路径
    folder_path = "./npyPaper2/srm_test"  # 修改为您的npy文件所在文件夹

    # 检查文件夹中的npy文件
    inspect_npy_folder(folder_path)


if __name__ == "__main__":
    main()