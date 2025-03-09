import colorsys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from .yolo_nets.yolo import YoloBody
from .utils_yolo.utils import cvtColor, get_classes, preprocess_input, resize_image
from .utils_yolo.utils_bbox import decode_outputs, non_max_suppression
from PIL import Image

import os #为了路径方便，之后连接成绝对路径
# 获取当前 Python 文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))


'''
训练自己的数据集必看注释！
'''
class YOLO_model(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : os.path.join(current_dir, "yolo_weights/yolo_tongue.pth"),
        "classes_path"      : os.path.join(current_dir, "classes/tongue_classes.txt"),
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [448, 448],
        #---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.7,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        # self.model_path=os.path.join(current_dir, "yolo_weights/yolo_tongue.pth")
        # self.classes_path=os.path.join(current_dir, "classes/tongue_classes.txt")
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        # show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def get_prompt(self, img):      
        #img是PIL图像，不是NUMPY数组，所以求大小之类的有不同
        image_shape = np.array(np.shape(img)[0:2])        
        # image_data  = resize_image(Image.fromarray(img),(448,448),False)    意义是什么？估计得到seg才能发现了
        image_data  = resize_image(img, (448,448),False)   
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()            
            outputs = self.net(images)                         
            outputs = decode_outputs(outputs, self.input_shape)            
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return None#没有检测出来舌头

            top_label   = np.array(results[0][:, 6], dtype = 'int32')            
            top_boxes   = results[0][:, :4]     
        if len(top_label)<1:
            return None#有边界框，但是没有对应的标签
        #得到框对应的坐标       
        box             = top_boxes[0]            
        top, left, bottom, right = box
        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))                      
        bottom  = min(img.size[1], np.floor(bottom).astype('int32'))
        right   = min(img.size[0], np.floor(right).astype('int32'))            
        result=np.array((left,top,right,bottom))                
        return result#舌头所在的框

    def generate(self, onnx=False):
        self.net    = YoloBody(self.num_classes, self.phi)
        device      = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # device      = torch.device('cpu')#在cpu上运行
        self.net.load_state_dict(torch.load(self.model_path, map_location=device,weights_only=True))
        self.net    = self.net.eval()
        
        
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, img, crop = False, count = False):
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        if isinstance(img, np.ndarray):
            image = Image.fromarray(img)
        else:
            image=img
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: #没有舌像
                return None

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]


        if len(top_label)<1:
            return None#有边界框，但是没有对应的标签
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#

        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            text_bbox = draw.textbbox((0, 0), label, font=font)
            label_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
            label = label.encode('utf-8')
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
    

    #---------------------------------------------------#
    #   检测单张图片,且不计数
    #---------------------------------------------------#
    def detect_single_image(self, img, crop = False):
        if isinstance(img, np.ndarray):
            image = Image.fromarray(img)
        else:
            image=img
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: #没有舌像
                return None,None,None,None

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]


        if len(top_label)<1:
            return None,None,None,None#有边界框，但是没有对应的标签
        
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#

        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))


        #---------------------------------------------------------#
        #   得到识别信息（类别，置信度，边界框）
        #---------------------------------------------------------#
        i=0
        c=top_label[i]

        predicted_class = self.class_names[int(c)]
        box             = top_boxes[i]
        score           = top_conf[i]

        top, left, bottom, right = box

        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
        right   = min(image.size[0], np.floor(right).astype('int32'))

        x_min=left
        y_min=top
        x_max=right
        y_max=bottom

        bbox=[x_min,y_min,x_max,y_max]#舌头的边界框

        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            crop_image = image.crop(bbox)

        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        # label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # draw.rectangle([left, top, right, bottom], outline=self.colors[c])
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            return image,bbox,score,crop_image
        else:
            return image,bbox,score,None