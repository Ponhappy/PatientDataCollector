import os
import time
import random
import cv2
import uuid
import requests
import numpy as np
from bs4 import BeautifulSoup
import urllib.parse
from tenacity import retry, stop_after_attempt, wait_fixed
from fake_useragent import UserAgent
from ultralytics import YOLO

# ================== 各分类的映射字典 ==================
COLOR_MAP = {
    "淡红舌": 0,
    "淡白舌": 1,
    "枯白舌": 2,
    "红舌": 3,
    "绛舌": 4,
    "舌尖红": 5,
    "舌边尖红": 6,
    "青紫舌": 7,
    "淡紫舌": 8,
    "瘀斑舌": 9,
    "瘀点舌": 10
}

SHAPE_MAP = {
    "老舌": 0,
    "淡嫩舌": 1,
    "红嫩舌": 2,
    "胖大舌": 3,
    "淡胖舌": 4,
    "肿胀舌": 5,
    "齿痕舌": 6,
    "红瘦舌": 7,
    "淡瘦舌": 8,
    "红点舌": 9,
    "芒刺舌": 10,
    "裂纹舌": 11,
    "淡裂舌": 12,
    "红裂舌": 13,
    "舌衄": 14,
    "舌疮": 15
}

COATING_COLOR_MAP = {
    "白苔": 0,
    "薄白润苔": 1,
    "薄白干苔": 2,
    "薄白滑苔": 3,
    "白厚苔": 4,
    "白厚腻苔": 5,
    "白厚腻干苔": 6,
    "积粉苔": 7,
    "白燥苔": 8,
    "黄苔": 9,
    "薄黄苔": 10,
    "深黄苔": 11,
    "焦黄苔": 12,
    "黄糙苔": 13,
    "黄滑苔": 14,
    "黄腻苔": 15,
    "黄黏腻苔": 16,
    "灰黑苔": 17,
    "灰黑腻润苔": 18,
    "灰黑干燥苔": 19,
    "相兼苔色": 20,
    "黄白相兼苔": 21,
    "黑（灰）白相兼苔": 22,
    "黄腻黑（灰）相兼苔": 23
}

COATING_TEXTURE_MAP = {
    "薄苔": 0,
    "厚苔": 1,
    "润苔": 2,
    "滑苔": 3,
    "燥苔": 4,
    "糙苔": 5,
    "腻苔": 6,
    "垢腻苔": 7,
    "黏腻苔": 8,
    "滑腻苔": 9,
    "燥腻苔": 10,
    "腐苔": 11,
    "脓腐苔": 12,
    "白霉苔": 13,
    "剥苔": 14,
    "淡剥苔": 15,
    "红剥苔": 16,
    "花剥苔": 17,
    "类剥苔": 18,
    "地图舌": 19,
    "镜面舌": 20,
    "镜面红舌": 21,
    "镜面淡舌": 22
}

CONDITION_MAP = {
    "正常舌态": 0,
    "舌歪斜": 1,
    "舌僵硬": 2,
    "舌痿软": 3,
    "舌短缩": 4,
    "舌吐弄": 5,
    "舌震颤": 6
}

STATUS_MAP = {
    "荣舌": 0,
    "枯舌": 1
}

VEIN_MAP = {
    "正常舌脉": 0,
    "舌脉粗长如网": 1,
    "舌脉曲张": 2,
    "舌脉瘀血": 3
}

# ================== 数据集配置 ==================
# 对应各大类的数据集名称（最终目录为 {dataset_name}dataset ）
DATASET_CONFIG = {
    "color": {"map": COLOR_MAP, "dir": "color_dataset"},
    "shape": {"map": SHAPE_MAP, "dir": "shape_dataset"},
    "coating_color": {"map": COATING_COLOR_MAP, "dir": "coating_color_dataset"},
    "coating_texture": {"map": COATING_TEXTURE_MAP, "dir": "coating_texture_dataset"},
    "condition": {"map": CONDITION_MAP, "dir": "condition_dataset"},
    "status": {"map": STATUS_MAP, "dir": "status_dataset"},
    "vein": {"map": VEIN_MAP, "dir": "vein_dataset"}
}

SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test
PAGES_PER_CATEGORY = 20
IMAGES_PER_PAGE = 40
DELAY_RANGE = (1, 4)

DETECTION_MODEL = YOLO("yolov8s.pt")
DETECTION_CONF = 0.4

BASE_DIR = "dataset"  # 基础目录

# ================== 核心功能类 ==================
class TongueDatasetBuilder:
    def __init__(self, dataset_key):
        # dataset_key 比如 "color", "shape", ...
        self.dataset_key = dataset_key
        self.category_map = DATASET_CONFIG[dataset_key]["map"]
        self.dataset_dir = DATASET_CONFIG[dataset_key]["dir"]
        self.ua = UserAgent()
        self.session = requests.Session()
        self._init_dirs()

    def _init_dirs(self):
        """初始化目录结构，比如：color_dataset/images/train, color_dataset/labels/train, etc."""
        for split in ['train', 'val', 'test']:
            for data_type in ['images', 'labels']:
                path = os.path.join(BASE_DIR, self.dataset_dir, data_type, split)
                for category in self.category_map.keys():
                    os.makedirs(os.path.join(path, category), exist_ok=True)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_url(self, url):
        headers = {
            "User-Agent": self.ua.random,
            "Referer": "https://www.bing.com/",
            "Accept-Language": "en-US,en;q=0.9"
        }
        return self.session.get(url, headers=headers, timeout=15)

    def _generate_split(self):
        r = random.random()
        if r < SPLIT_RATIOS[0]:
            return 'train'
        elif r < sum(SPLIT_RATIOS[:2]):
            return 'val'
        else:
            return 'test'

    def _detect_tongue(self, img_path, category):
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None

            results = DETECTION_MODEL(img)
            boxes = []

            for result in results:
                for box in result.boxes:
                    if box.conf.item() > DETECTION_CONF:
                        x_center = (box.xyxyn[0][0] + box.xyxyn[0][2]) / 2
                        y_center = (box.xyxyn[0][1] + box.xyxyn[0][3]) / 2
                        width = box.xyxyn[0][2] - box.xyxyn[0][0]
                        height = box.xyxyn[0][3] - box.xyxyn[0][1]
                        # 这里简单检查面积阈值（可根据需要调整）
                        if width * height > 0.05:
                            # 通过当前 dataset 的映射获取类别ID
                            category_id = self.category_map.get(category, None)
                            if category_id is not None:
                                boxes.append([
                                    category_id,
                                    x_center.item(),
                                    y_center.item(),
                                    width.item(),
                                    height.item()
                                ])
            return boxes if boxes else None
        except Exception as e:
            print(f"检测异常: {str(e)}")
            return None

    def _delete_temp_file(self, path):
        if os.path.exists(path):
            os.remove(path)

    def process_image(self, content, category):
        temp_path = None
        try:
            unique_id = uuid.uuid4().hex
            temp_path = f"temp_{unique_id}.jpg"
            with open(temp_path, "wb") as f:
                f.write(content)

            boxes = self._detect_tongue(temp_path, category)
            if not boxes:
                print(f"图片无效或检测失败: {temp_path}")
                return False

            split = self._generate_split()
            # 图片和标签保存到对应的大类数据集中
            img_dir = os.path.join(BASE_DIR, self.dataset_dir, "images", split, category)
            label_dir = os.path.join(BASE_DIR, self.dataset_dir, "labels", split, category)

            img_path = os.path.join(img_dir, f"{unique_id}.jpg")
            label_path = os.path.join(label_dir, f"{unique_id}.txt")

            os.rename(temp_path, img_path)
            np.savetxt(label_path, boxes, fmt="%d %.6f %.6f %.6f %.6f")
            print(f"保存 {category} 图像和标签到 {split} 集")
            return True
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                self._delete_temp_file(temp_path)

# ================== 爬虫引擎 ==================
class BingCrawler:
    def __init__(self, dataset_key):
        self.dataset_builder = TongueDatasetBuilder(dataset_key)

    def _extract_image_urls(self, html_content):
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
        print(f"\n开始处理类别: {category}")
        for page in range(PAGES_PER_CATEGORY):
            print(f"处理第 {page + 1}/{PAGES_PER_CATEGORY} 页...")
            url = f"https://www.bing.com/images/search?q={urllib.parse.quote(category)}&first={page * IMAGES_PER_PAGE}"
            try:
                response = self.dataset_builder._fetch_url(url)
                if not response.ok:
                    continue

                image_urls = self._extract_image_urls(response.text)
                for img_url in image_urls:
                    try:
                        img_response = self.dataset_builder._fetch_url(img_url)
                        if img_response.status_code != 200:
                            continue

                        if self.dataset_builder.process_image(img_response.content, category):
                            print(f"成功入库: {category} - {img_url[-15:]}")
                        time.sleep(random.uniform(*DELAY_RANGE))
                    except Exception as e:
                        print(f"图片处理异常: {str(e)}")
            except Exception as e:
                print(f"页面处理异常: {str(e)}")

# ================== 主程序 ==================
if __name__ == "__main__":
    # 遍历所有大类
    for dataset_key, config in DATASET_CONFIG.items():
        print(f"\n开始爬取 {dataset_key} 数据集，共 {len(config['map'])} 个类别")
        for category in config['map'].keys():
            print(f"\n【{dataset_key}】当前爬取类别: {category}")
            crawler = BingCrawler(dataset_key)
            crawler.crawl_category(category)
    print("所有数据集构建完成！")

