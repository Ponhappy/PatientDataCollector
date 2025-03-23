import matplotlib.pyplot as plt
from PIL import ImageFont

import numpy as np
import cv2
# from utils.SkinUtils import *
# from core.FaceLandMark import faceDetection


def cs(titile="0x123", img=None):
    cv2.imshow(titile, img)
    cv2.waitKey(0)
    cv2.destroyWindow(titile)


chi_sample = cv2.imread("face_diagnose_model/four_color_face_sample/chi/ting_trim.jpg")
black_sample = cv2.imread("face_diagnose_model/four_color_face_sample/black/ting_trim.jpg")
white_sample = cv2.imread("face_diagnose_model/four_color_face_sample/white/ting_trim.jpg")
yellow_sample = cv2.imread("face_diagnose_model/four_color_face_sample/yellow/ting_trim.jpg")


# img_predict_roi_ting = cv2.resize(img_predict_roi_ting, img_sample_roi_ting.shape[::-1][1:3])

def getDistance1(predict, sample):
    """
    去掉黑色像素，这是被肤色检测处理过的像素
    :param predict:
    :param sample:
    :return:
    """
    x = predict.shape[0]
    y = predict.shape[1]
    dist_byloop = []
    for i in range(x):
        for j in range(y):
            # 纯黑色不检测，因为这是被肤色检测处理过的像素
            # if (predict[i][j] == (0, 0, 0)).all() or (sample[i][j] == (0, 0, 0)).all():
            if (predict[i][j] == (0, 0, 0)).all():
                continue

            A = predict[i][j]
            B = sample[i][j]
            # np.linalg.norm(A - B) 等同于
            # np.sqrt(np.sum((A[0] - B[0])**2 + (A[1] - B[1])**2 +(A[2] - B[2])**2))
            dist_byloop.append(np.linalg.norm(A - B))
            # sum += np.sqrt(np.sum(np.square(predict[i][j] - sample[i][j])))
    return np.mean(dist_byloop)


def getDistance2ByLab(predict, sample):
    """
    :param predict:
    :param sample:
    :return:
    """

    def trimBlack(img_lab):
        k = 0
        L, A, B = 0, 0, 0
        for row in img_lab:
            for v in row:
                # 排除黑色
                if v[0] != 0:
                    k = k + 1
                    L += v[0]
                    A += v[1]
                    B += v[2]
        # 计算出了LAB的均值
        L0 = int(round(L / k))
        A0 = int(round(A / k))
        B0 = int(round(B / k))
        return L0, A0, B0

    predict_lab = cv2.cvtColor(predict, cv2.COLOR_BGR2Lab)
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2Lab)
    pl, pa, pb = trimBlack(predict_lab)
    sl, sa, sb = trimBlack(sample_lab)

    # distance = ((pa - sa) ** 2 + (pb - sb) ** 2) ** 0.5
    # distance = ((pl - sl) ** 2 + (pa - sa) ** 2 + (pb - sb) ** 2)
    distance = ((pa - sa) ** 2 + (pb - sb) ** 2)
    return distance


def getDistance2BHSV(predict, sample):
    """
    :param predict:
    :param sample:
    :return:
    """

    def trimBlack(img_hsv):
        k = 0
        H, S, V = 0, 0, 0
        for row in img_hsv:
            for v in row:
                # 排除黑色
                if v[0] != 0:
                    k = k + 1
                    H += v[0]
                    S += v[1]
                    V += v[2]
        # 计算出了LAB的均值
        H0 = int(round(H / k))
        S0 = int(round(S / k))
        V0 = int(round(V / k))
        return H0, S0, V0

    predict_hsv = cv2.cvtColor(predict, cv2.COLOR_BGR2HSV)
    sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    ph, ps, pv = trimBlack(predict_hsv)
    sh, ss, sv = trimBlack(sample_hsv)
    # distance = ((ph - sh) ** 2 + (ps - ss) ** 2 + (pv - sv) ** 2)
    distance = (ph - sh) ** 2
    return distance


def getDistanceYCrCb(predict, sample):
    """
    :param predict:
    :param sample:
    :return:
    """

    def trimBlack(img_hsv):
        k = 0
        Y, Cr, Cb = 0, 0, 0
        for row in img_hsv:
            for v in row:
                # 排除黑色
                if v[0] != 0:
                    k = k + 1
                Y += v[0]
                Cr += v[1]
                Cb += v[2]
        # 计算出了LAB的均值
        H0 = int(round(Y / k))
        S0 = int(round(Cr / k))
        V0 = int(round(Cb / k))
        return H0, S0, V0

    predict_YCrCb = cv2.cvtColor(predict, cv2.COLOR_BGR2YCrCb)
    sample_YCrCb = cv2.cvtColor(sample, cv2.COLOR_BGR2YCrCb)
    pY, pCr, pCb = trimBlack(predict_YCrCb)
    sY, sCr, sCb = trimBlack(sample_YCrCb)
    # distance = ((pY - sY) ** 2 + (pCr - sCr) ** 2 + (pCb - sCb) ** 2)
    distance = ((pCr - sCr) ** 2 + (pCb - sCb) ** 2)
    return distance


def getDistanceByRGB(predict, sample):
    def trimBlack(img):
        k = 0
        B, G, R = 0, 0, 0
        for row in img:
            for v in row:
                # 排除黑色
                # if v[0] != 0:
                if v[0] != 0 and v[1] != 0 and v[2] != 0:
                    k = k + 1
                    R += v[0]
                    G += v[1]
                    R += v[2]
        # 计算出了LAB的均值
        R0 = int(round(R / k))
        G0 = int(round(G / k))
        B0 = int(round(R / k))
        return R0, G0, B0

    # predict_lab = cv2.cvtColor(predict, cv2.COLOR_BGR)
    # sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2Lab)
    pr, pg, pb = trimBlack(predict)
    sr, sg, sb = trimBlack(sample)

    # distance = ((pa - sa) ** 2 + (pb - sb) ** 2) ** 0.5
    distance = ((pr - sr) ** 2 + (pg - sg) ** 2 + (pb - sb) ** 2)
    return distance


def distance(input_path, sample=None):
    predict = cv2.imread(input_path)
    predict = cv2.resize(predict, (sample.shape[1], sample.shape[0]))
    res = np.hstack([predict, sample])
    # cv2.imshow(name, res)
    return getDistance1(predict, sample), getDistance2ByLab(predict, sample), getDistance2BHSV(predict,
           sample), getDistanceByRGB(predict, sample), getDistanceYCrCb(predict, sample)

def description(str,color):
    if str=="庭":
        if color == "红": des = "印堂呈现自然淡红，是心气充沛、心血充盈、血脉运行通畅的表现。"
        if color == "白": des = "可能近期劳累、休息不足，经调整作息后多可恢复。"
        if color == "黑": des = "若伴有轻微失眠，可能是心肾不交的早期表现，通过调整作息、睡前避免过度兴奋，有助于缓解。"
        if color == "黄": des = "若伴有轻微心悸，可能是心脾功能稍不协调，通过调节饮食和情绪，可逐渐改善。"

    if str=="左颊":
        if color == "红": des = "左脸颊微微泛红且色泽均匀，反映肝血充足，肝气条达舒畅。"
        if color == "白": des = "可能是近期用眼过度或情绪稍显焦虑，影响了肝脏的疏泄功能。"
        if color == "黑": des = "左脸颊颜色偏黑但不暗沉，若无明显不适，可能是近期熬夜较多，影响了肝脏的排毒功能。"
        if color == "黄": des = "若无胁肋部不适，可能是近期情绪波动影响了肝脏气血运行。"

    if str=="鼻":
        if color == "红": des = "皮肤状态良好，提示脾胃运化功能正常。"
        if color == "白": des = "可能是脾胃功能稍弱，但处于可自我调节范围。"
        if color == "黑": des = "可能是脾胃受寒邪侵袭，阳气受损。"
        if color == "黄": des = "可能是脾胃运化功能在适应季节变化或饮食调整过程中出现的暂时现象。"

    if str=="右颊":
        if color == "红": des = "右脸颊呈现淡淡的红润，表明肺气充足，肺的宣发与肃降功能正常。"
        if color == "白": des = "若呼吸平稳、无咳嗽等症状，可能是肺卫功能稍弱，对外部环境变化的适应能力稍差。"
        if color == "黑": des = "可能是肺部受到外界环境中不良因素的轻微影响。"
        if color == "黄": des = "肺的宣发功能稍有不足，导致水液代谢在面部的表现稍有异常。"

    if str=="颌":
        if color == "红": des = "下颌部位略带淡红，说明肾中精气充足，阴阳平衡。"
        if color == "白": des = "若日常小便正常，无腰膝酸软等症状，可能是肾的阳气稍不足。"
        if color == "黑": des = "膝无明显酸软疼痛，可能是近期过度劳累、体力消耗较大，影响了肾脏功能。"
        if color == "黄": des = "若无尿频、尿急等症状，可能是肾脏对水液代谢的调节功能在自我调整。"
    return(des)


def skin_color_detection(input_path, face_label,color_analysis_dir):
    dt_chi = distance(input_path, chi_sample)
    dt_black = distance(input_path, black_sample)
    dt_white = distance(input_path, white_sample)
    dt_yellow = distance(input_path, yellow_sample)

    font = ImageFont.truetype("../../fonts/simsun.ttc", 10)

    a = np.vstack([dt_chi, dt_black, dt_white, dt_yellow])

    # b = 1 - (a / a.sum(axis=0, keepdims=1))
    b = a / a.sum(axis=0, keepdims=1)
    # x = ['chi', 'black', 'white', 'yellow']
    x = ['赤', '黑', '白', '黄']
    # x = np.asarray([1,2,3,4])

    # plt.figure(figsize=(100, 50), dpi=8)
    plt.ylim(0, 1)
    plt.title("预测值")
    plt.xlabel('four color')
    plt.ylabel('probability')
    # $正则$
    # plt.yticks([0.2, 0.4, 0.6, 0.8], ['r$very\ bad$', r'$bad$', 'normal', 'good'])
    plt.yticks([0.2, 0.4, 0.6, 0.8])

    barwidth = 0.15
    x1 = list(range(len(x)))  # [0,1,2,3]
    y1 = b.transpose()[:1][0]  # 取矩阵第一列[0.xxx, 0.xx, 0.xx, 0.xx]
    x2 = [i + barwidth for i in x1]
    y2 = b.transpose()[1:2][0]
    x3 = [i + barwidth * 2 for i in x1]
    y3 = b.transpose()[2:3][0]
    x4 = [i + barwidth * 3 for i in x1]
    y4 = b.transpose()[3:4][0]

    x5 = [i + barwidth * 4 for i in x1]
    y5 = b.transpose()[4:5][0]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置x轴刻度
    plt.xticks(x2, x)

    for xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5, in zip(x1, x2, x3, x4, x5, y1, y2, y3, y4, y5):
        # ha: horizontal alignment
        plt.text(xx1, yy1 + 0.04, '%.2f' % yy1, ha='center', va='top')
        plt.text(xx2, yy2 + 0.04, '%.2f' % yy2, ha='center', va='top')
        plt.text(xx3, yy3 + 0.04, '%.2f' % yy3, ha='center', va='top')
        plt.text(xx4, yy4 + 0.04, '%.2f' % yy4, ha='center', va='top')
        plt.text(xx5, yy5 + 0.04, '%.2f' % yy5, ha='center', va='top')

    y_all = list(y3+y4+y5)
    index = y_all.index(min(y_all))
    if index == 0: color = "红"
    if index == 1: color = "黑"
    if index == 2: color = "白"
    if index == 3: color = "黄"
    # 防止des未定义的情况
    try:
        des = description(face_label, color)
        print(des)
    except Exception as e:
        print(f"错误: {e}")
        des = "未能获取描述"
    
    print(index)

    plt.bar(x1, y1, label='Euclidean Distance', width=barwidth)
    plt.bar(x2, y2, label='Lab distance', width=barwidth)
    plt.bar(x3, y3, label='HSV', width=barwidth)
    plt.bar(x4, y4, label='RGB', width=barwidth)
    plt.bar(x5, y5, label='YCrCb', width=barwidth)
    plt.legend()
    plt.title(face_label)
    # plt.show() # 移除这行，防止启动新的事件循环
    
    # 保存图片前确保目录存在
    output_path = color_analysis_dir
    plt.savefig(output_path + '/' + face_label + '.png')
    plt.close() # 添加这行关闭图形

    return(color)
    # 散点图：
    """
    横坐标:Cr [0,255]
    纵坐标:Cb [0,255]
    中心点：
    对应关系：先画出横坐标，再画出纵坐标
    """
if __name__ == '__main__':
    index = skin_color_detection("four_color_face_sample/yellow/ming_tang_trim.jpg", "庭")
    print("该面部部位的颜色判断为：", index)