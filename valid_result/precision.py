import pandas as pd
import numpy as np
dates = pd.read_csv('predictions.txt',header=None)
def check_target_in_prediction(data):
    prediction = data[0]
    target = data[1][12:-3]
    prediction = prediction[16:16+len(target)]
    #print(target)
    # 使用 in 运算符检查字符串包含关系
    is_target_in_prediction = target in prediction

    return is_target_in_prediction

# 你提供的 JSON 数据

# 打印结果
n=0
for i in range(len(dates)):
    s=check_target_in_prediction(dates.iloc[i])
    if s==True:
        n=n+1
print(n)