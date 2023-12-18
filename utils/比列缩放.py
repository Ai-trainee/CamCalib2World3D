from PIL import Image, ImageFilter, ImageDraw, ImageFont
import os

def resize_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            #如果out不存在则创建
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, filename)
            with Image.open(input_path) as img:
                width, height = img.size
                if width == height:  # 检查宽高比是否为1:1
                    # new_img = img.resize((640, 640), Image.ANTIALIAS)
                    new_img = img.resize((1580, 1580), Image.LANCZOS)
                    new_img.save(output_path)
                    print(f"{filename} 已缩放到 640x640 像素，并保存到 {output_path}。")
                else:
                    print(f"{filename} 的宽高比不是1:1，未进行缩放。")

input_folder = r"F:\picss3"  # 请替换为你的输入文件夹路径
output_folder = r"F:\picss3\1580"  # 请替换为你的输出文件夹路径
resize_images(input_folder, output_folder)

