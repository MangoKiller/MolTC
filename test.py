import os

def count_subdirectories(folder_path):
    try:
        # 获取文件夹下的所有文件和子文件夹名
        entries = os.listdir(folder_path)

        # 过滤出子文件夹
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]

        # 返回子文件夹的数量
        return len(subdirectories)
    except FileNotFoundError:
        print(f"文件夹 '{folder_path}' 不存在。")
        return -1  # 返回 -1 表示文件夹不存在
    except Exception as e:
        print(f"发生错误：{e}")
        return -2  # 返回 -2 表示发生了其他错误
print(count_subdirectories("/home/fangjf/git-code/data/ddi_data/drugbank/valid/text"))