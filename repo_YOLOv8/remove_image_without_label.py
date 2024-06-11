import os
import shutil

src_label = 'E:/bdd100k/labels/train'
src_image = 'E:/bdd100k/images/100k/train'
dest = 'E:/bdd100k/images/100k/image_without_label'
label_list = os.listdir(src_label)

for i in os.listdir(src_image):
    file_name = i.split('.')[0] + '.txt'
    if file_name not in label_list:
        shutil.move(os.path.join(src_image, i), os.path.join(dest, i))
