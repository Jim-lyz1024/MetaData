import os

# 指定图片所在的文件夹路径
folder_path = r"E:\LYZ\AucklandCourse\2024Thesis\Metadata\stoat\auxiliary_network_pics"
output_file = r"E:\LYZ\AucklandCourse\2024Thesis\Metadata\stoat\image_names.txt"

# 获取文件夹下的所有文件，并过滤出以 .JPG 结尾的文件
image_names = [f for f in os.listdir(folder_path) if f.endswith('.JPG')]

# 将图片名字写入 .txt 文件
with open(output_file, 'w') as f:
    for image_name in image_names:
        f.write(image_name + ' \n')

print(f"Image file created: {output_file}")
