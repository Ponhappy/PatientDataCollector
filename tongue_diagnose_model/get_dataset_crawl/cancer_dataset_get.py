import os
import time
import random
import cv2
import uuid
import requests
import numpy as np
from bs4 import BeautifulSoup
import urllib.parse
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from fake_useragent import UserAgent
from ultralytics import YOLO

# ================== 全局配置 ==================
##舌头病变
CATEGORY_MAP = {
    "舌糜": 0,
    "舌疮": 1,
    "重舌": 2,
    "舌菌（舌癌）": 3,
    "舌菌（舌乳头状瘤）": 4,
    "舌血管瘤": 5,
    "舌烂": 6,
    "舌衄": 7,
    "正常舌头": 8
}



BASE_DIR = "../dataset/cancer_dataset"
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test
PAGES_PER_CATEGORY = 10
IMAGES_PER_PAGE = 35
DELAY_RANGE = (1, 4)

DETECTION_MODEL = YOLO("yolov8s.pt")
DETECTION_CONF = 0.2


# ================== 核心功能类 ==================
class TongueDatasetBuilder:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self._init_dirs()

    def _init_dirs(self):
        """初始化目录结构"""
        for split in ['train', 'val', 'test']:
            for data_type in ['images', 'labels']:
                for category in CATEGORY_MAP:
                    path = os.path.join(BASE_DIR, data_type, split, category)
                    os.makedirs(path, exist_ok=True)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_url(self, url):
        """带重试机制的请求"""
        headers = {
            "User-Agent": self.ua.random,
            "Referer": "https://www.bing.com/",
            "Accept-Language": "en-US,en;q=0.9"
        }
        return self.session.get(url, headers=headers, timeout=15)

    def _generate_split(self):
        """动态数据划分"""
        r = random.random()
        if r < SPLIT_RATIOS[0]: return 'train'
        if r < sum(SPLIT_RATIOS[:2]): return 'val'
        return 'test'

    def _detect_tongue(self, img_path, category):
        """舌体检测与标注生成"""
        try:
            img = cv2.imread(img_path)
            if img is None: return None

            results = DETECTION_MODEL(img)
            boxes = []

            for result in results:
                for box in result.boxes:
                    # 输出检测框的详细信息，帮助调试
                    print(f"检测框：类别ID={box.cls.item()}, 置信度={box.conf.item()}, 坐标={box.xywh}")

                    if box.conf.item() > DETECTION_CONF:  # 置信度大于设定的阈值
                        x_center = (box.xyxyn[0][0] + box.xyxyn[0][2]) / 2
                        y_center = (box.xyxyn[0][1] + box.xyxyn[0][3]) / 2
                        width = box.xyxyn[0][2] - box.xyxyn[0][0]
                        height = box.xyxyn[0][3] - box.xyxyn[0][1]

                        # 检查检测框面积，如果面积过小，可能是无效框
                        if width * height > 0.05:  # 可以适当降低阈值
                            category_id = CATEGORY_MAP.get(category, None)
                            if category_id is not None:
                                boxes.append([
                                    category_id,  # 动态类别ID
                                    x_center.item(),
                                    y_center.item(),
                                    width.item(),
                                    height.item()
                                ])
                            else:
                                print(f"警告：未能找到类别 '{category}' 对应的ID")
            return boxes if boxes else None
        except Exception as e:
            print(f"检测异常: {str(e)}")
            return None

    def _delete_temp_file(self, path):
        """删除临时文件"""
        if os.path.exists(path):
            os.remove(path)

    def process_image(self, content, category):
        """完整的图片处理流水线"""
        temp_path = None
        try:
            # 临时存储
            unique_id = uuid.uuid4().hex
            temp_path = f"temp_{unique_id}.jpg"
            with open(temp_path, "wb") as f:
                f.write(content)

            # 执行检测
            boxes = self._detect_tongue(temp_path, category)
            if not boxes:
                print(f"图片无效或检测失败: {temp_path}")
                return False

            # 确定存储路径
            split = self._generate_split()
            img_dir = os.path.join(BASE_DIR, "images", split, category)
            label_dir = os.path.join(BASE_DIR, "labels", split, category)

            # 持久化存储
            img_path = os.path.join(img_dir, f"{unique_id}.jpg")
            label_path = os.path.join(label_dir, f"{unique_id}.txt")

            os.rename(temp_path, img_path)

            # 保存标签时，确保正确输出类别ID
            print(f"保存标签到: {label_path}, 类别ID: {CATEGORY_MAP[category]}")
            np.savetxt(label_path, boxes, fmt="%d %.6f %.6f %.6f %.6f")

            return True
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return False
        finally:
            if temp_path:
                self._delete_temp_file(temp_path)


# ================== 爬虫引擎 ==================
class BingCrawler:
    def __init__(self):
        self.dataset_builder = TongueDatasetBuilder()

    def _extract_image_urls(self, html_content):
        """HTML内容解析"""
        soup = BeautifulSoup(html_content, "html.parser")
        urls = []

        for elem in soup.find_all(["img", "a"]):
            src = None
            if elem.name == "img":
                src = elem.get("data-src") or elem.get("src")
            elif elem.name == "a" and "iusc" in elem.get("class", []):
                try:
                    meta = urllib.parse.parse_qs(elem.get("m", ""))
                    src = meta.get("murl", [None])[0]
                except:
                    pass

            if src and "th?id=OIP." not in src:
                urls.append(src)

        return list(set(urls))

    def crawl_category(self, category):
        """单类别爬取流程"""
        print(f"\n开始处理类别: {category}")

        for page in range(PAGES_PER_CATEGORY):
            print(f"处理第 {page + 1}/{PAGES_PER_CATEGORY} 页...")
            url = f"https://www.bing.com/images/search?q={urllib.parse.quote(category)}&first={page * IMAGES_PER_PAGE}"

            try:
                response = self.dataset_builder._fetch_url(url)
                if not response.ok: continue

                image_urls = self._extract_image_urls(response.text)
                for img_url in image_urls:
                    try:
                        img_response = self.dataset_builder._fetch_url(img_url)
                        if img_response.status_code != 200: continue

                        if self.dataset_builder.process_image(img_response.content, category):
                            print(f"成功入库: {category}/{img_url[-15:]}")

                        time.sleep(random.uniform(*DELAY_RANGE))
                    except Exception as e:
                        print(f"图片处理异常: {str(e)}")
            except Exception as e:
                print(f"页面处理异常: {str(e)}")


# ================== 主程序 ==================
if __name__ == "__main__":
    crawler = BingCrawler()
    for category in CATEGORY_MAP:
        crawler.crawl_category(category)
    print("数据集构建完成！")
