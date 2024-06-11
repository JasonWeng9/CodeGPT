import os

src_image_train = 'E:/bdd100k/images/train'
src_image_val = 'E:/bdd100k/images/val'
src_image_test = 'E:/bdd100k/images/test'


with open('./data/bdd_train.txt', 'a') as f1:
    for i in os.listdir(src_image_train):
        f1.write(src_image_train + '/' + i + '\n')

with open('./data/bdd_val.txt', 'a') as f2:
    for i in os.listdir(src_image_val):
        f2.write(src_image_val + '/' + i + '\n')

with open('./data/bdd_test.txt', 'a') as f3:
    for i in os.listdir(src_image_test):
        f3.write(src_image_test + '/' + i+ '\n')