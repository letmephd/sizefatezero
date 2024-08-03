import cv2
import numpy as np
import os

def extract_mask(image_path):
    """
    从图像中提取mask图像。
    图像中非纯黑的部分将被转换为白色mask。
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # 检查图像是否读取成功
    if image is None:
        print(f"无法读取图像 {image_path}")
        return None
    
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 创建一个mask图像，所有像素初始为黑色
    mask = np.zeros_like(gray_image)
    
    # 将非纯黑的部分设置为白色
    mask[gray_image > 0] = 255
    
    return mask

def process_images(input_dir, output_dir):
    """
    处理指定目录下的所有图像，提取mask并保存。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        # 检查文件是否为图像文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 提取mask
            mask = extract_mask(file_path)
            
            if mask is not None:
                # 保存mask图像
                mask_filename = os.path.splitext(filename)[0] + '_mask.png'
                mask_path = os.path.join(output_dir, mask_filename)
                cv2.imwrite(mask_path, mask)
                print(f"保存mask图像: {mask_path}")

# 示例用法
input_directory = '/home/yinzijin/experiments/gaojiayi/FatezeroDragon/FateZero/data/C19_swan_small/raw_object'  # 输入图像文件夹路径
output_directory = '/home/yinzijin/experiments/gaojiayi/FatezeroDragon/FateZero/data/C19_swan_small/raw_object_mask'  # 输出mask文件夹路径
process_images(input_directory, output_directory)