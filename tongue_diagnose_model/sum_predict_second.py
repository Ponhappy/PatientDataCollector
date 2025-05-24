import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import datetime
import os

# ================== 配置中心 ==================
CLASS_MAPS = {
    "舌色": {
        "淡红舌": 0, "淡白舌": 1, "枯白舌": 2, "红舌": 3,
        "绛舌": 4, "舌尖红": 5, "舌边尖红": 6, "青紫舌": 7,
        "淡紫舌": 8, "瘀斑舌": 9, "瘀点舌": 10
    },
    "舌形": {
        "老舌": 0, "淡嫩舌": 1, "红嫩舌": 2, "胖大舌": 3,
        "淡胖舌": 4, "肿胀舌": 5, "齿痕舌": 6, "红瘦舌": 7,
        "淡瘦舌": 8, "红点舌": 9, "芒刺舌": 10, "裂纹舌": 11,
        "淡裂舌": 12, "红裂舌": 13, "舌衄": 14, "舌疮": 15
    },
    "苔色": {
        "白苔": 0, "薄白润苔": 1, "薄白干苔": 2, "薄白滑苔": 3,
        "白厚苔": 4, "白厚腻苔": 5, "白厚腻干苔": 6, "积粉苔": 7,
        "白燥苔": 8, "黄苔": 9, "薄黄苔": 10, "深黄苔": 11,
        "焦黄苔": 12, "黄糙苔": 13, "黄滑苔": 14, "黄腻苔": 15,
        "黄黏腻苔": 16, "灰黑苔": 17, "灰黑腻润苔": 18,
        "灰黑干燥苔": 19, "相兼苔色": 20, "黄白相兼苔": 21,
        "黑（灰）白相兼苔": 22, "黄腻黑（灰）相兼苔": 23
    },
    "苔质": {
        "薄苔": 0, "厚苔": 1, "润苔": 2, "滑苔": 3, "燥苔": 4,
        "糙苔": 5, "腻苔": 6, "垢腻苔": 7, "黏腻苔": 8, "滑腻苔": 9,
        "燥腻苔": 10, "腐苔": 11, "脓腐苔": 12, "白霉苔": 13,
        "剥苔": 14, "淡剥苔": 15, "红剥苔": 16, "花剥苔": 17,
        "类剥苔": 18, "地图舌": 19, "镜面舌": 20, "镜面红舌": 21,
        "镜面淡舌": 22
    },
    "舌态": {
        "正常舌态": 0, "舌歪斜": 1, "舌僵硬": 2, "舌痿软": 3,
        "舌短缩": 4, "舌吐弄": 5, "舌震颤": 6
    },
    "舌神": {"荣舌": 0, "枯舌": 1},
    "舌脉": {
        "正常舌脉": 0, "舌脉粗长如网": 1,
        "舌脉曲张": 2, "舌脉瘀血": 3
    }
}

MODEL_PATHS = {
    "detection": "yolov8s.pt",
    "舌色": "tongue_diagnose_model/runs/runs_color/train_results/weights/best.pt",
    "舌形": "tongue_diagnose_model/runs/runs_shape/train_results/weights/best.pt",
    "苔色": "tongue_diagnose_model/runs/runs_coating_color/train_results/weights/best.pt",
    "苔质": "tongue_diagnose_model/runs/runs_coating_texture/train_results/weights/best.pt",
    "舌态": "tongue_diagnose_model/runs/runs_condition/train_results/weights/best.pt",
    "舌神": "tongue_diagnose_model/runs/runs_status/train_results/weights/best.pt",
    "舌脉": "tongue_diagnose_model/runs/runs_vein/train_results/weights/best.pt"
}

# ================== 诊断知识库 ==================
# 各维度的详细诊断，每一类别对应一种诊断结果
DIAGNOSIS_RULES = {
    "舌色": {
        "淡红舌": {"证型": "正常舌色", "分析": "气血调和，脏腑功能正常。"},
        "淡白舌": {"证型": "气血两虚证", "分析": "阳气不足，血液生化乏源，常伴畏寒乏力。"},
        "枯白舌": {"证型": "血虚极证", "分析": "血虚严重，常伴面色苍白、心悸、头晕。"},
        "红舌": {"证型": "实热证", "分析": "内热明显，热盛血涌，可能伴有烦躁、口干。"},
        "绛舌": {"证型": "阴虚火旺", "分析": "热邪入营，阴液亏损，常伴有失眠、口渴。"},
        "舌尖红": {"证型": "心火亢盛", "分析": "心火旺盛，常伴情绪波动、心烦失眠。"},
        "舌边尖红": {"证型": "肝胆火旺", "分析": "肝火上炎，易伴口苦、易怒。"},
        "青紫舌": {"证型": "血瘀证", "分析": "血液循环受阻，常伴局部疼痛或麻木。"},
        "淡紫舌": {"证型": "气滞血瘀", "分析": "气机不畅，血流缓慢，可能伴胸闷。"},
        "瘀斑舌": {"证型": "重度血瘀", "分析": "明显的血瘀表现，提示局部或全身血液循环障碍。"},
        "瘀点舌": {"证型": "轻度血瘀", "分析": "局部血瘀轻微，需注意改善循环。"}
    },
    "舌形": {
        "老舌": {"证型": "体虚老态", "分析": "长期虚劳或慢性病后期，体内气血不足。"},
        "淡嫩舌": {"证型": "体质平和", "分析": "舌体柔嫩，状态较佳。"},
        "红嫩舌": {"证型": "阴虚火旺", "分析": "津液亏损，体内热邪明显。"},
        "胖大舌": {"证型": "脾虚湿盛", "分析": "脾失健运，水湿内停。"},
        "淡胖舌": {"证型": "阳虚水湿", "分析": "阳气不足，水液停聚，体质偏寒。"},
        "肿胀舌": {"证型": "湿毒内蕴", "分析": "湿热凝聚明显，舌体肿胀。"},
        "齿痕舌": {"证型": "脾虚证", "分析": "脾气虚弱，水湿内停，常见于消化不良。"},
        "红瘦舌": {"证型": "阴虚火旺", "分析": "阴液亏损，火热内扰，体质消瘦。"},
        "淡瘦舌": {"证型": "气血两虚", "分析": "气血不足，体质消瘦，常伴乏力。"},
        "红点舌": {"证型": "热毒初起", "分析": "局部热象初现，提示热邪开始入侵。"},
        "芒刺舌": {"证型": "热盛伤津", "分析": "热邪猛烈，损伤津液，易出现局部不适。"},
        "裂纹舌": {"证型": "阴液亏虚", "分析": "津液不足，舌面干燥，有裂纹出现。"},
        "淡裂舌": {"证型": "轻度阴虚", "分析": "轻度津液不足，需适当滋阴。"},
        "红裂舌": {"证型": "阴虚火旺", "分析": "阴液亏损严重，内热明显。"},
        "舌衄": {"证型": "血热妄行", "分析": "血热过盛，易出血，需警惕内热异常。"},
        "舌疮": {"证型": "实热证", "分析": "热毒内盛，局部溃疡，常伴口腔疼痛。"}
    },
    "苔色": {
        "白苔": {"证型": "寒证", "分析": "外感风寒或脾胃虚寒，体内寒邪较重。"},
        "薄白润苔": {"证型": "正常苔色", "分析": "显示胃气充盈、津液充足。"},
        "薄白干苔": {"证型": "轻度阴虚", "分析": "轻微津液不足，内热初起。"},
        "薄白滑苔": {"证型": "寒湿偏盛", "分析": "提示水湿较重，但尚未转热。"},
        "白厚苔": {"证型": "脾虚湿困", "分析": "脾气虚弱，水湿内停。"},
        "白厚腻苔": {"证型": "寒湿困脾", "分析": "体内寒湿较重，脾运失调。"},
        "白厚腻干苔": {"证型": "重寒湿证", "分析": "津液严重不足，寒湿内停。"},
        "积粉苔": {"证型": "痰湿停滞", "分析": "痰湿凝聚，运化功能障碍。"},
        "白燥苔": {"证型": "阴虚火旺", "分析": "津液亏损明显，体内热邪偏重。"},
        "黄苔": {"证型": "湿热证", "分析": "内热与湿邪交织，体内热邪较盛。"},
        "薄黄苔": {"证型": "轻度湿热", "分析": "湿热初起，提示体内热偏重。"},
        "深黄苔": {"证型": "湿热较盛", "分析": "体内湿热明显，舌苔较厚。"},
        "焦黄苔": {"证型": "热毒炽盛", "分析": "热邪炽盛，津液枯竭。"},
        "黄糙苔": {"证型": "湿热偏盛", "分析": "体内湿热较重，舌面粗糙。"},
        "黄滑苔": {"证型": "湿热偏盛", "分析": "湿热内蕴，津液不足。"},
        "黄腻苔": {"证型": "湿热内蕴", "分析": "湿热积聚明显，痰浊较重。"},
        "黄黏腻苔": {"证型": "重湿热证", "分析": "湿热较重，易伴内分泌失调。"},
        "灰黑苔": {"证型": "寒湿积聚", "分析": "寒湿内停，正气不足。"},
        "灰黑腻润苔": {"证型": "寒湿偏盛", "分析": "寒湿较重，但尚有一定润泽。"},
        "灰黑干燥苔": {"证型": "极重寒湿", "分析": "寒湿凝滞明显，津液耗损。"},
        "相兼苔色": {"证型": "内外兼治", "分析": "寒热兼杂或多邪混合。"},
        "黄白相兼苔": {"证型": "湿热兼寒", "分析": "表里不一，内热外寒。"},
        "黑（灰）白相兼苔": {"证型": "血瘀寒凝", "分析": "血液循环受阻，同时寒湿内停。"},
        "黄腻黑（灰）相兼苔": {"证型": "重湿热兼血瘀", "分析": "湿热与血瘀并存，病情复杂。"}
    },
    "舌脉": {
        "正常舌脉": {"证型": "舌脉正常", "分析": "气血运行良好。"},
        "舌脉粗长如网": {"证型": "气血瘀滞", "分析": "脉络明显，可能存在血瘀。"},
        "舌脉曲张": {"证型": "血瘀阻滞", "分析": "脉络曲张，提示血液循环不畅。"},
        "舌脉瘀血": {"证型": "重度血瘀", "分析": "血瘀明显，需注意改善血流。"}
    },
    "苔质": {
        "薄苔": {"证型": "正常苔质", "分析": "胃气充盈，津液未伤，正常生理现象。"},
        "厚苔": {"证型": "邪盛正实", "分析": "病邪内盛，正气未虚，常见于实证。"},
        "腻苔": {"证型": "湿浊内阻", "分析": "湿浊或痰饮内阻，脾胃运化失常。"},
        "腐苔": {"证型": "食积胃热", "分析": "食积化热，胃中腐熟功能失常，常伴有口臭。"},
        "剥苔": {"证型": "胃阴亏虚", "分析": "胃阴不足，舌苔部分或全部脱落，常见于慢性胃病。"},
        "镜面红舌": {"证型": "胃阴枯竭", "分析": "胃阴耗竭，舌面光滑如镜，提示严重胃阴亏损。"}
    },
    "舌态": {
        "痿软舌": {"证型": "气血两虚", "分析": "气血亏虚，舌体软弱无力，难以伸展。"},
        "强硬舌": {"证型": "热入心包", "分析": "热邪炽盛，侵入心包，舌体强直，言语不清。"},
        "震颤舌": {"证型": "肝风内动", "分析": "肝风内动，舌体震颤，常见于高热或惊厥。"},
        "歪斜舌": {"证型": "中风先兆", "分析": "脉络阻滞，舌体偏斜，提示中风或中风先兆。"},
        "短缩舌": {"证型": "寒凝筋脉", "分析": "寒邪凝滞，筋脉收缩，舌体短缩难伸。"},
        "吐舌": {"证型": "心脾有热", "分析": "心脾蕴热，舌频频吐出，常见于小儿高热。"},
        "弄舌": {"证型": "心脾热盛", "分析": "心脾积热，舌不停搅动，常见于小儿智障或热病。"}
    },
    "舌神": {
        "荣舌": {"证型": "正气未衰", "分析": "舌色红活润泽，活动自如，反映机体正气充足，预后良好。"},
        "枯舌": {"证型": "正气亏虚", "分析": "舌色晦暗枯涩，活动不灵，提示机体正气衰退，病情较重，预后不佳。"}
    }
}


COMBO_RULES = {
    ("淡白舌", "齿痕舌"): {
        "证型": "脾虚湿盛",
        "治法": "健脾利湿",
        "方剂": "参苓白术散"
    },
    ("红舌", "黄腻苔"): {
        "证型": "肝胆湿热",
        "治法": "清利湿热",
        "方剂": "龙胆泻肝汤"
    },
    ("淡白舌", "胖大舌", "白厚腻苔"): {
        "证型": "脾肾阳虚，水湿内停",
        "治法": "温阳利水",
        "方剂": "真武汤"
    },
    ("红舌", "裂纹舌", "少苔"): {
        "证型": "阴虚内热",
        "治法": "滋阴清热",
        "方剂": "知柏地黄丸"
    },
    ("绛舌", "瘀点舌", "黄厚苔"): {
        "证型": "热入营血，血瘀内阻",
        "治法": "清营凉血，化瘀通络",
        "方剂": "清营汤合血府逐瘀汤"
    },
    ("青紫舌", "瘀斑舌", "舌脉曲张"): {
        "证型": "气滞血瘀",
        "治法": "理气活血",
        "方剂": "血府逐瘀汤"
    },
    ("淡红舌", "裂纹舌", "薄白苔"): {
        "证型": "脾胃阴虚",
        "治法": "养阴益胃",
        "方剂": "益胃汤"
    },
    ("红舌", "芒刺舌", "黄厚苔"): {
        "证型": "脾胃湿热",
        "治法": "清热燥湿",
        "方剂": "葛根芩连汤"
    },
    ("淡白舌", "齿痕舌", "白厚腻苔"): {
        "证型": "脾虚湿困",
        "治法": "健脾利湿",
        "方剂": "参苓白术散"
    },
    ("红舌", "红点舌", "黄腻苔"): {
        "证型": "湿热蕴结",
        "治法": "清热利湿",
        "方剂": "茵陈蒿汤"
    },
    ("绛舌", "裂纹舌", "少苔"): {
        "证型": "阴虚火旺",
        "治法": "滋阴降火",
        "方剂": "大补阴丸"
    },
    ("淡紫舌", "瘀斑舌", "白厚苔"): {
        "证型": "寒凝血瘀",
        "治法": "温经散寒，活血化瘀",
        "方剂": "当归四逆汤"
    },
    ("红舌", "胖大舌", "黄厚腻苔"): {
        "证型": "脾胃湿热",
        "治法": "清热利湿",
        "方剂": "甘露消毒丹"
    },
    ("淡白舌", "淡嫩舌", "薄白苔"): {
        "证型": "气血两虚",
        "治法": "益气养血",
        "方剂": "八珍汤"
    },
    ("红舌", "裂纹舌", "黄燥苔"): {
        "证型": "热盛伤津",
        "治法": "清热生津",
        "方剂": "白虎汤"
    },
    ("绛舌", "瘀点舌", "无苔"): {
        "证型": "阴虚血瘀",
        "治法": "滋阴活血",
        "方剂": "通幽汤"
    },
    ("青紫舌", "瘀斑舌", "厚腻苔"): {
        "证型": "痰瘀互结",
        "治法": "化痰祛瘀",
        "方剂": "二陈汤合血府逐瘀汤"
    },
    ("淡红舌", "裂纹舌", "薄黄苔"): {
        "证型": "气阴两虚",
        "治法": "益气养阴",
        "方剂": "生脉散"
    },
    ("红舌", "芒刺舌", "黄厚腻苔"): {
        "证型": "湿热内蕴",
        "治法": "清热化湿",
        "方剂": "三仁汤"
    },
    ("淡白舌", "齿痕舌", "白厚腻苔", "舌脉曲张"): {
        "证型": "脾虚湿盛，血瘀阻络",
        "治法": "健脾利湿，活血化瘀",
        "方剂": "参苓白术散合血府逐瘀汤"
    }
}



# ================== 核心服务 ==================
class TongueAnalysisSystem:
    def __init__(self):
        self.models = self._load_models()
        self.detector = YOLO(MODEL_PATHS["detection"])

    def _load_models(self):
        """加载所有分类模型"""
        models = {}
        for cat, path in MODEL_PATHS.items():
            if cat == "detection":
                continue
            try:
                if not os.path.exists(path):
                    print(f"错误: 模型文件不存在: {path}")
                    continue
                print(f"正在加载模型: {cat} (路径: {path})")
                models[cat] = YOLO(path)
                print(f"成功加载模型: {cat}")
            except Exception as e:
                print(f"错误: 无法加载模型 {cat}: {str(e)}")
        return models

    def predict_category(self, img_path, category):
        """执行分类预测"""
        try:
            # 首先检查模型是否存在
            if category not in self.models:
                print(f"警告: 类别 '{category}' 的模型未加载")
                return "未检出"
            
            model = self.models[category]
            
            # 检查图像是否存在
            if not os.path.exists(img_path):
                print(f"警告: 图像文件不存在: {img_path}")
                return "未检出"
            
            # 执行预测
            results = model.predict(img_path, conf=0.3)  # 降低置信度阈值
            
            # 如果检测到结果
            if results[0].boxes and len(results[0].boxes.cls) > 0:
                class_id = int(results[0].boxes.cls[0].item())
                class_name = next((k for k, v in CLASS_MAPS[category].items() if v == class_id), None)
                if class_name:
                    print(f"成功检测 {category}: {class_name} (置信度: {results[0].boxes.conf[0].item():.2f})")
                    return class_name
                else:
                    print(f"警告: 类别ID {class_id} 在 {category} 类别映射中未找到")
            else:
                print(f"警告: {category} 类别没有检测到任何对象")
            
            return "未检出"
        except Exception as e:
            print(f"错误: 预测 {category} 时出现问题: {str(e)}")
            import traceback
            traceback.print_exc()
            return "未检出"

    def detect_tongue(self, img_path):
        """检测舌头区域并标注"""
        results = self.detector.predict(img_path)
        if results[0].boxes:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            draw.rectangle(box.tolist(), outline="red", width=3)
            return img, box
        return None, None


# ================== 诊断引擎 ==================
class DiagnosisEngine:
    @staticmethod
    def analyze(features):
        """生成结构化诊断报告"""
        reports = {}

        # 各维度独立分析
        for category in ["舌色", "舌形", "苔色", "舌脉","苔质","舌态","舌神"]:
            value = features.get(category)
            if value in DIAGNOSIS_RULES.get(category, {}):
                reports[category] = DIAGNOSIS_RULES[category][value]

        feature_tuple = tuple(
            features[category] for category in ["舌色", "舌形", "苔色", "苔质", "舌态", "舌神", "舌脉"])
        if feature_tuple in COMBO_RULES:
            reports["组合特征"] = COMBO_RULES[feature_tuple]
        # 生成自然语言描述
        return DiagnosisEngine._format_report(reports, features)

    @staticmethod
    def _format_report(reports, features):
        """格式化诊断结果"""
        diagnosis = []
        treatment = []

        # 基础特征
        diagnosis.append("【基础特征分析】")
        for cat in ["舌色", "舌形", "苔色", "苔质", "舌态", "舌神", "舌脉"]:
            diagnosis.append(f"{cat}: {features[cat]}")

        # 证型分析
        diagnosis.append("\n【中医辨证】")
        for cat in reports:
            if cat in DIAGNOSIS_RULES:
                info = reports[cat]
                diagnosis.append(f"{cat} → {info['证型']}: {info['分析']}")

        # 组合证型
        if "组合特征" in reports:
            combo = reports["组合特征"]
            diagnosis.append(f"\n复合证型: {combo['证型']}")
            treatment.append(f"推荐方剂: {combo['方剂']}")

        return "\n".join(diagnosis), "\n".join(treatment)


# ================== 报告生成 ==================
class ReportGenerator:
    @staticmethod
    def generate_html_report(image_path, features, diagnosis, treatment,
                                 boxes=None, output_html="tongue_report.html"):
            """
            生成包含多维度特征标注的HTML报告
            features包含7类特征：
                - 舌色、舌形、苔色、苔质、舌态、舌神、舌脉
            boxes字典包含检测框信息（可选）
            """
            # 图像预处理
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("无法加载图像文件")

            # 转换为Pillow图像用于标注
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # 设置中文字体
            try:
                font = ImageFont.truetype("simsun.ttc", 24, encoding="utf-8")
            except IOError:
                print("警告：使用默认字体")
                font = ImageFont.load_default()

            # 在图像上标注特征信息
            y_offset = 30
            feature_colors = {
                '舌色': (0, 255, 0),  # 绿色
                '舌形': (255, 0, 0),  # 蓝色
                '苔色': (0, 0, 255),  # 红色
                '苔质': (255, 255, 0),  # 青色
                '舌态': (0, 255, 255),  # 黄色
                '舌神': (128, 0, 128),  # 紫色
                '舌脉': (255, 165, 0)  # 橙色
            }

            # 绘制检测框（如果存在）
            if boxes and boxes.get('tongue'):
                x1, y1, x2, y2 = boxes['tongue']
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=3)

            # 添加特征标注
            for category in ['舌色', '舌形', '苔色', '苔质', '舌态', '舌神', '舌脉']:
                text = f"{category}: {features.get(category, 'N/A')}"
                draw.text((10, y_offset), text, font=font, fill=feature_colors[category])
                y_offset += 30  # 每行间隔30像素

            # 保存标注图像
            annotated_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite("annotated.jpg", annotated_img)

            # 生成HTML内容
            html_content = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <title>多维舌诊报告</title>
                <style>
                    .container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                    .features {{ background: #f0f8ff; padding: 20px; border-radius: 10px; }}
                    .diagnosis {{ background: #fff0f5; padding: 20px; border-radius: 10px; }}
                    img {{ max-width: 100%; border: 2px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>多维舌诊分析报告</h1>
                <div class="container">
                    <div class="features">
                        <h2>特征检测结果</h2>
                        <img src="annotated.jpg">
                        <ul>
                            {"".join(f"<li><b>{k}</b>: {v}</li>" for k, v in features.items())}
                        </ul>
                    </div>

                    <div class="diagnosis">
                        <h2>中医辨证</h2>
                        <div style="white-space: pre-wrap;">{diagnosis}</div>

                        <h2>综合调理方案</h2>
                        <div style="white-space: pre-wrap;">{treatment}</div>

                        <h3>风险提示</h3>
                        <p style="color:{"red" if features.get('舌神') == '枯舌' else "green"}">
                            {"⚠️ 癌症高风险提示" if features.get('舌神') == '枯舌' else "✅ 未发现高风险特征"}
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """

            with open(output_html, "w", encoding="utf-8") as f:
                f.write(html_content)

            return output_html

# ================== 主函数 ==================
def sum_predict(test_image):
    system = TongueAnalysisSystem()

    # 特征提取
    features = {
        "舌色": system.predict_category(test_image, "舌色"),
        "舌形": system.predict_category(test_image, "舌形"),
        "苔色": system.predict_category(test_image, "苔色"),
        "苔质": system.predict_category(test_image, "苔质"),
        "舌态": system.predict_category(test_image, "舌态"),
        "舌神": system.predict_category(test_image, "舌神"),
        "舌脉": system.predict_category(test_image, "舌脉")
    }

    # 舌头检测
    annotated_img, tongue_box = system.detect_tongue(test_image)
    features["tongue_box"] = tongue_box.tolist() if tongue_box is not None else []

    # 生成诊断
    diagnosis, treatment = DiagnosisEngine.analyze(features)

    # 生成报告
    report_path = ReportGenerator.generate_html_report(
        test_image, features, diagnosis, treatment
    )

    return (
        features["苔质"], features["舌态"], features["舌神"], features["舌脉"],features,
        report_path, diagnosis, treatment
    )

if __name__ == "__main__":
    # 执行诊断
    result = sum_predict("img_3.png")

    # 输出结果
    print("=== 基础特征 ===")
    print(result[-2])

    print("\n=== 调理建议 ===")
    print(result[-1])

    print(f"\n报告路径: {result[-3]}")