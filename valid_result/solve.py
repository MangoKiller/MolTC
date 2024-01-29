import json
import math
# 读取文本文件按行
def accuratecy():
    print(1)
    import json
    file_path = 'valid_result/predictions.txt'  # 替换为您的文本文件路径
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"文件 '{file_path}' 未找到.")
        lines = []
    import re

    # 遍历每一行并尝试解析JSON
    valid_num = 0
    valid_rmse = 0
    test_num = 0
    test_rmse = 0
    #再此处划分验证集和测试集
    mid = len(lines)/2
    m=0
    for line in lines:
        if m<mid:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"行解析错误：{e}")
                data = None

            # 如果成功解析JSON，则提取预测值和目标值
            if data:
                prediction = data.get('prediction', '')
                target = data.get('target', '')
                prediction_numbers = re.findall(r'-?\d+\.\d+', prediction)
                p1=float(prediction_numbers[-1])
                target_numbers = re.findall(r'-?\d+\.\d+', target)
                p2=float(target_numbers[-1])
                valid_num = valid_num +abs(p1-p2)
                valid_rmse = valid_rmse +math.pow(p1-p2, 2)
        else:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"行解析错误：{e}")
                data = None

            # 如果成功解析JSON，则提取预测值和目标值
            if data:
                prediction = data.get('prediction', '')
                target = data.get('target', '')
                prediction_numbers = re.findall(r'-?\d+\.\d+', prediction)
                p1=float(prediction_numbers[-1])
                target_numbers = re.findall(r'-?\d+\.\d+', target)
                p2=float(target_numbers[-1])
                test_num = test_num +abs(p1-p2)
                test_rmse = test_rmse +math.pow(p1-p2, 2)
        m=m+1
    valid_p=valid_num/mid
    valid_rmse = math.sqrt(valid_rmse/mid)
    test_p=test_num/(len(lines)-mid)
    test_rmse = math.sqrt(test_rmse/(len(lines)-mid))



    file_path = 'valid_result/predictions.txt'

    # 打开文件并清空内容
    with open(file_path, 'w') as file:
        file.truncate(0)

    print(f'文件 "{file_path}" 已清空并保存。')



    file_path = 'valid_result/your_file.txt'

    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 在最后一行写入一个数字（假设这个数字是42）
    lines.append("验证集MAE:"+str(valid_p)+".RMSE:"+str(valid_rmse)+". 测试集MAE:"+str(test_p)+".RMSE:"+str(test_rmse) + '\n')

    # 打开文件并写入修改后的内容
    with open(file_path, 'w') as file:
        file.writelines(lines)