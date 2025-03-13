# import cv2

# def check_camera(camera_id=1):
#     # 创建VideoCapture对象
#     cap = cv2.VideoCapture(camera_id)
    
#     # 检查摄像头是否成功打开
#     if cap.isOpened():
#         print("摄像头已打开")
#         # 释放摄像头资源
#         cap.release()
#     else:
#         print("摄像头未打开")

# # 检测默认摄像头（一般为0）
# check_camera()


import cv2

def show_camera(camera_id=1):
    # 创建VideoCapture对象，打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按'q'键退出...")
    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        
        # 如果正确读取帧，ret为True
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 显示结果帧
        cv2.imshow('摄像头画面', frame)
        
        # 按'q'键退出循环
        if cv2.waitKey(1) == ord('q'):
            break
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

# 显示默认摄像头画面（一般为0）
show_camera()