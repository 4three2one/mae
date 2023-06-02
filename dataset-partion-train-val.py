import glob


import os
import shutil
import random

from tqdm import  tqdm


source="/home/wjxy/Downloads/tomota_leaf_csdn/*"
root=os.path.split(source)[0]
target_train=f"{root}_tomato_tv/train"
target_val=f"{root}_tomato_tv/val"

ratio=0.8

# 获取一个子目录中所有的图片文件
extensions = ['jpg', 'jpeg','JPG','JPEG'] # 注意下这里

def main():
    if not os.path.exists(target_train):
        os.makedirs(target_train)
    if not os.path.exists(target_val):
        os.makedirs(target_val)
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(source, '*.' + extension)
        print(file_glob)
        file_list.extend(glob.glob(file_glob))
    print(len(file_list))
    random.shuffle(file_list)
    l=len(file_list)
    train=int(ratio*l)
    files_train=file_list[:train]
    files_val=file_list[train:]
    base_num=100000
    for index,file in tqdm(enumerate(files_train)):
        # filename = os.path.basename(file).replace('JPG', 'jpg')
        path=os.path.join(target_train,file.split('/')[-2])
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(file,os.path.join(path,str(base_num+index)+'.jpg'))
    base_num = 200000
    for index,file in tqdm(enumerate(files_val)):
        path = os.path.join(target_val, file.split('/')[-2])
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(file,os.path.join(path,str(base_num+index)+'.jpg'))
    pass


if __name__ == '__main__':
    main()