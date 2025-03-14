from picseg import pic_seg
from color_distance import skin_color_detection
import os
import cv2

# 定义图片名称到字符的映射
name_mapping = {
    'jia_left_roi': '左颊',
    'jia_right_roi': '右颊',
    'ke_roi': '颌',
    'ming_tang_roi': '鼻',
    'ting_roi': '庭'

}

def main(input_path):
    print("正在裁剪图片...")
    pic_seg(input_path)
    print("裁剪完成！")
    print("开始进行面诊分析...")
    folder_path = 'faceseg/roi_images'
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 构建图片的完整路径
            image_path = os.path.join(folder_path, filename)
            # 读取图片
            image = cv2.imread(image_path)
            if image is not None:
                # 从文件名中提取关键部分（去除扩展名）
                name_key = os.path.splitext(filename)[0]
                # 根据映射关系获取对应的字符
                result_char = str(name_mapping.get(name_key))
                if result_char:
                    print(f"图片 {filename} 对应的字符是: {result_char}")
                    # 调用分析函数对图片进行分析
                    skin_color_detection(image_path, result_char)
                else:
                    print(f"未找到图片 {filename} 对应的字符映射。")
            else:
                print(f"无法读取图片 {filename}。")

if __name__ == '__main__':
    main(input_path = "four_color_face_sample/black.png" )