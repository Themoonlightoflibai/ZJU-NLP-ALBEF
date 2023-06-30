import json


if __name__ == '__main__':
    # 打开JSON文件
    with open('tinytest.json', 'r') as file:
        # 解析JSON数据
        data = json.load(file)

    cnt = 0
    # 遍历每个字典
    for item in data:
        # 处理当前字典
        print(item)
        cnt +=1
        if cnt ==10:
            break