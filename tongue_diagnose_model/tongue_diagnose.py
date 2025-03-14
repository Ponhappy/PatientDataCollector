import os
import cv2
import base64
import numpy as np
from ultralytics import YOLO
from .predict import cancer_predict
from .predict import coating_predict
from .predict import color_predict

# ---------------- 全局配置：加载舌头检测模型 ----------------
# 用于检测图片中是否存在舌头的模型（YOLOv8s）
DETECTION_MODEL = YOLO("yolov8s.pt")
DETECTION_CONF = 0.25  # 舌头检测置信度阈值

# ---------------- 舌苔、舌病变、舌头颜色类别映射 ----------------
COATING_CATEGORY_MAP = {
    "薄白润苔": 0,  # 正常胃气充盈，津液充沛
    "薄白滑苔": 1,  # 水湿偏重，提示阳虚不运
    "白腻苔": 2,
    "白滑腻苔": 3,
    "白粘腻苔": 4,
    "白厚腻苔": 5,
    "白厚松苔": 6,
    "白腻苔化燥": 7,
    "黄腻苔化燥": 8,
    "薄黄苔": 9,
    "黄腻苔": 10,
    "黄粘腻苔": 11,
    "黄滑苔": 12,
    "黄燥苔": 13,
    "黄瓣苔": 14,
    "灰黄腻苔": 16,
    "焦黑苔": 19,
    "黑燥苔": 20,
    "剥苔": 21,
    "地图舌": 24,
    "镜面苔": 25,
    "无根苔": 26,
    "薄白苔和黄厚腻苔": 27,
    "黄厚粘腻苔与薄白苔": 28,
    "正常舌": 29
}

CANCER_CATEGORY_MAP = {
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

COLOR_CATEGORY_MAP = {
    "正常舌": 0,
    "淡白舌": 1,
    "枯白舌": 2,
    "红舌": 3,
    "舌尖红": 4,
    "红绛舌": 5,
    "青紫舌": 6,
    "淡紫舌": 7,
    "淡紫淤堵舌": 8,
    "紫红舌": 9,
    "绛紫舌": 10,
    "瘀斑舌": 11
}

# ---------------- 详细舌苔调理细则及建议 ----------------
COATING_DETAIL = {
    "薄白润苔": {
        "diagnosis": "舌苔薄白而润",
        "analysis": "显示胃气充盈、津液充足，体内阴阳平衡。",
        "diet": "保持五色饮食，多摄入新鲜蔬果及优质蛋白；避免暴饮暴食。",
        "exercise": "每天至少30分钟中等强度运动，如快走、游泳、太极拳。"
    },
    "薄白滑苔": {
        "diagnosis": "舌苔薄白滑腻",
        "analysis": "提示体内水湿偏重，可能与阳虚不运有关。",
        "diet": "宜温补阳气，少吃寒凉生冷，多食生姜、桂圆、红枣等温性食材。",
        "exercise": "推荐太极拳、八段锦或散步，每天20-30分钟。"
    },
    "白腻苔": {
        "diagnosis": "舌苔白腻",
        "analysis": "反映湿浊、痰饮或食滞，常见于脾胃虚弱或消化不良。",
        "diet": "建议清淡饮食，减少油腻、甜腻食品，多吃山药、薏仁、苦瓜。",
        "exercise": "适量慢跑或快步走，每天30分钟。"
    },
    "白厚腻苔": {
        "diagnosis": "舌苔厚腻",
        "analysis": "提示脾胃虚弱，湿浊内停，痰湿积聚较重。",
        "diet": "宜温热易消化，如红豆粥、薏仁粥；避免寒凉食物。",
        "exercise": "每天30分钟快走或慢跑。"
    },
    "剥苔": {
        "diagnosis": "舌苔剥落",
        "analysis": "提示胃阴或胃气亏虚，津液不足，多见于体虚或长期劳累者。",
        "diet": "宜滋阴养胃，推荐银耳百合粥、鲜枣莲子粥；忌辛辣刺激和过冷食物。",
        "exercise": "建议轻松散步和太极拳，每天约20分钟。"
    },
    "地图舌": {
        "diagnosis": "地图舌",
        "analysis": "表现为舌面剥落呈不规则地图状，多见于胃气或胃阴不足，亦与过敏体质有关。",
        "diet": "宜补充营养，推荐鸡汤、红枣山药粥；记录食物日记，注意过敏原。",
        "exercise": "建议进行舒缓有氧运动，如慢跑或瑜伽，每天30分钟。"
    },
    "镜面苔": {
        "diagnosis": "镜面舌",
        "analysis": "舌面光滑如镜，提示胃气、胃阴极度亏损，见于热病伤阴后期或严重虚劳。",
        "diet": "宜滋阴补液，多食海参粥、绿豆百合粥；避免辛辣油炸。",
        "exercise": "建议以静为主，进行轻柔伸展和呼吸训练，每天15-20分钟。"
    },
    "无根苔": {
        "diagnosis": "无根苔",
        "analysis": "提示胃气大伤、胃阴极亏，病情较严重，常伴其他器官功能紊乱。",
        "diet": "宜温补流质食物，如粥类和温补汤品，严禁寒凉刺激。",
        "exercise": "建议以休息为主，仅进行非常轻柔的体操和呼吸训练。"
    }
    # 其他类型可按需要增加
}

# ---------------- 组合调理增强建议 ----------------
COMBO_ADVICE = {
    ("淡白舌", "白厚腻苔"): {
        "阶段调理": {
            "第一阶段(1-2周)": {
                "重点": "温阳化湿",
                "breakfast": "姜丝小米粥+炒白扁豆",
                "exercise": "上午10点晒太阳15分钟",
                "diet": "早餐宜温热易消化，避免生冷。"
            },
            "第二阶段(3-4周)": {
                "重点": "气血双补",
                "食疗": "当归黄芪乌鸡汤（每周2次）",
                "massage": "足三里穴艾灸（每周3次）",
                "exercise": "结合适量步行，促进气血运行。",
                "diet": "均衡营养，多食红枣、桂圆。"
            }
        }
    },
    ("红舌", "黄燥苔"): {
        "三日应急方案": {
            "Day1": "西瓜汁200ml（分次小口含服）",
            "Day2": "梨藕百合羹（慢炖2小时）",
            "Day3": "石斛麦冬茶（石斛5g+麦冬10g）"
        },
        "禁忌提示": "治疗期间禁止汗蒸、桑拿，避免剧烈运动。"
    },
    ("夏季", "黄腻苔"): {
        "特色调理": {
            "解暑饮品": "三豆饮（黑豆、绿豆、赤小豆各15g）",
            "外用法": "佩兰香囊（佩兰10g+薄荷5g）随身佩戴"
        }
    },
    ("冬季", "淡白舌"): {
        "温补方案": {
            "药膳": "羊肉板栗煲（加当归5g）",
            "泡脚": "艾叶30g+花椒10g煮水泡脚"
        }
    }
}

# ---------------- 舌苔改善进展评估 ----------------
class CoatingProgress:
    def __init__(self):
        self.stage_records = []

    def evaluate_improvement(self, previous, current):
        improvement_rules = {
            ("黄腻苔", "薄黄苔"): "湿热渐退，体内热邪减轻",
            ("白厚腻苔", "薄白润苔"): "脾运恢复，气血趋于平稳",
            ("镜面苔", "剥苔"): "胃阴渐复，体内阴液改善"
        }
        return improvement_rules.get((previous, current), "暂无明显变化，继续观察")

# ---------------- 增加舌头检测：调用YOLOv8s检测舌头区域 ----------------
def detect_tongue_box(image_path, conf_threshold=DETECTION_CONF):
    """
    使用YOLOv8s模型检测舌头区域，返回带检测框的图片及第一条检测框坐标（格式：(x1, y1, x2, y2)）。
    """
    results = DETECTION_MODEL.predict(source=image_path, conf=conf_threshold, show=False)
    if results and len(results) > 0:
        annotated_img = results[0].plot()
        if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy()  # 格式为 [x1, y1, x2, y2]
            return annotated_img, box
    print("未检测到舌头区域！")
    return None, None

# ---------------- 舌诊辨证模块（结合三模型预测结果） ----------------
def tongue_diagnose_sum(test_image):
    """
    结合舌色、舌苔及舌病变三个模型的预测结果，
    得出综合详细的中医舌诊报告，并生成包含各检测框（含舌头检测框）的 HTML 报告。
    """

    # 处理舌头颜色预测
    color = color_predict.detect_and_predict_color(test_image)
    # 处理舌苔预测
    coating = coating_predict.detect_and_predict_coating(test_image)
    # 处理舌病变预测
    cancer = cancer_predict.detect_and_predict_cancer(test_image)

    # 生成检测报告描述文字
    color_report = f"舌色预测结果：{color}"
    coating_report = f"舌苔预测结果：{coating}"
    cancer_report = f"舌病变预测结果：{cancer}"

    # 进行辨证并生成调理建议，扩充诊断结果
    diagnosis, treatment = diagnose_tongue(color, coating, cancer)
    if coating in COATING_DETAIL:
        detail = COATING_DETAIL[coating]
        treatment += (f"\n\n【详细舌苔调理建议】\n诊断：{detail['diagnosis']}\n"
                      f"分析：{detail['analysis']}\n饮食建议：{detail['diet']}\n锻炼建议：{detail['exercise']}")

    combo_key = (color, coating)
    if combo_key in COMBO_ADVICE:
        combo_info = COMBO_ADVICE[combo_key]
        if "阶段调理" in combo_info:
            treatment += "\n\n【阶段调理方案】\n" + str(combo_info["阶段调理"])
        if "三日应急方案" in combo_info:
            treatment += "\n\n【三日应急方案】\n" + str(combo_info["三日应急方案"])

    # 获取舌头检测框（调用YOLOv8s检测）
    tongue_annotated, tongue_box = detect_tongue_box(test_image)
    # 组合所有检测框信息
    boxes = {
        'tongue': tongue_box  # 新增舌头检测框
    }

    # 生成 HTML 报告，包含所有检测框标注的图片和详细诊断说明
    image_pth=generate_html_report(test_image, color, coating, cancer, boxes, diagnosis, treatment)

    # 返回三个特征的检测结果，标框图以及检测报告描述文字
    return color_report,coating_report,cancer_report,image_pth,diagnosis,treatment


def diagnose_tongue(color, coating, cancer):
    # 扩充诊断：详细描述症候、可能病因及建议
    diagnosis_rules = [
        {
            "condition": lambda c, t, s: c == "正常舌" and t == "薄白润苔" and s == "正常舌头",
            "diagnosis": "舌象正常",
            "treatment": "舌色、舌苔及舌部形态均正常，无明显内外邪气。建议维持现有健康生活方式，并定期体检。"
        },
        {
            "condition": lambda c, t, s: c == "淡白舌" and t == "薄白润苔",
            "diagnosis": "气血两虚证",
            "treatment": "可能存在气血不足，表现为面色苍白、精神不振。建议改善饮食、适当温补，增强体质。"
        },
        {
            "condition": lambda c, t, s: c == "淡白舌" and t == "白滑腻苔",
            "diagnosis": "脾虚湿困证",
            "treatment": "脾胃功能减弱，湿气内生。建议健脾祛湿，多摄入山药、薏仁等健脾食材，并加强运动。"
        },
        {
            "condition": lambda c, t, s: c == "枯白舌",
            "diagnosis": "精气衰败证",
            "treatment": "体虚明显，伴有疲劳、消瘦等症状。建议温补气血，增加营养摄入，并注意充分休息。"
        },
        {
            "condition": lambda c, t, s: c == "舌尖红",
            "diagnosis": "心火上炎证",
            "treatment": "舌尖红提示心火亢盛，可能伴烦躁、失眠。建议清心降火，调整情绪，配合中药调理。"
        },
        {
            "condition": lambda c, t, s: c in ["红舌", "红绛舌"] and t in ["黄腻苔", "黄粘腻苔"],
            "diagnosis": "湿热内蕴证",
            "treatment": "湿热内蕴，可能伴口干、便秘等症状。建议清热利湿、调理脾胃，保持清淡饮食。"
        },
        {
            "condition": lambda c, t, s: c == "青紫舌" and t == "白厚腻苔",
            "diagnosis": "痰瘀互结证",
            "treatment": "血液循环受阻，可能伴有胸闷、疼痛。建议活血化瘀、化痰通络，并适当中药调理。"
        },
        {
            "condition": lambda c, t, s: c == "紫红舌" and t == "焦黑苔",
            "diagnosis": "热毒炽盛证",
            "treatment": "热毒内盛，可能导致高热、口渴。建议清热解毒，多饮水，必要时尽快就医。"
        },
        {
            "condition": lambda c, t, s: c == "淡紫舌" and t == "灰黄腻苔",
            "diagnosis": "寒热错杂证",
            "treatment": "体内寒热失调，可能伴寒热往来、精神不振。建议调和阴阳，适当进补与清热。"
        },
        {
            "condition": lambda c, t, s: c == "淡紫淤堵舌",
            "diagnosis": "寒凝血瘀证",
            "treatment": "血瘀阻滞明显，可能伴有痛经、瘀血。建议活血化瘀、温经调理，必要时中医辨证治疗。"
        }
    ]
    for rule in diagnosis_rules:
        if rule["condition"](color, coating, cancer):
            if cancer != "正常舌头":
                return enhance_diagnosis(rule, cancer)
            diagnosis_text = rule["diagnosis"] + "【扩展诊断】：结合检测数据和临床表现，建议密切关注体征变化，定期复查。"
            treatment_text = rule["treatment"] + "\n【扩展建议】：改善生活习惯、合理饮食、适度运动；若症状持续或加重，请及时就医。"
            return diagnosis_text, treatment_text
    # 通用诊断及建议
    return generic_diagnosis(color, coating, cancer)

def enhance_diagnosis(base_rule, cancer):
    cancer_enhancements = {
        "舌糜": (
            f"{base_rule['diagnosis']}伴局部舌糜",
            f"{base_rule['treatment']}\n【健康提示】：舌糜可能预示慢性疾病，建议详细检查，并在医生指导下调理。"
        ),
        "瘀斑舌": (
            f"{base_rule['diagnosis']}兼血瘀证",
            f"{base_rule['treatment']}\n【健康提示】：舌部瘀斑提示血液循环障碍，建议中医辨证调理，改善血流。"
        ),
        "舌衄": (
            f"{base_rule['diagnosis']}伴动血异常",
            f"{base_rule['treatment']}\n【健康提示】：舌衄可能由内热或虚火引起，建议尽快就医查明原因。"
        )
    }
    diag, treat = cancer_enhancements.get(
        cancer,
        (
            f"{base_rule['diagnosis']}（伴局部病变）",
            f"{base_rule['treatment']}\n【健康提示】：检测到舌部异常，建议及时就医明确病因。"
        )
    )
    treat += "\n【扩展诊断】：鉴于检测到病变，建议咨询专业医师，进行详细检查以明确治疗方案。"
    return diag, treat

def generic_diagnosis(color, coating, cancer):
    color_map = {
        "淡白舌": (
            "气血虚", "建议多摄入高营养食物，如红枣、黑芝麻、鸡蛋等，适当温补以增强体质。", "每天早晨进行30分钟慢跑或快走。"
        ),
        "红舌": (
            "内热偏盛", "建议清热解毒，避免辛辣油炸，多饮绿豆汤或苦瓜汁。", "每天散步30分钟，避免剧烈运动。"
        ),
        "青紫舌": (
            "血瘀阻滞", "建议活血化瘀，适量进补具有活血功能的食物。", "建议练习太极拳或气功，每天20分钟。"
        ),
        "淡紫舌": (
            "阳虚血瘀", "建议温阳活血，多摄入温补食材，如核桃、桂圆、红枣。", "每日快步走30分钟，并结合适当拉伸。"
        )
    }
    coating_map = {
        "薄白润苔": (
            "表证正常", "显示胃气充盈、津液充足，建议饮食均衡，多摄入新鲜蔬果。", "保持每日30分钟适度运动。"
        ),
        "黄腻苔": (
            "湿热内蕴", "提示湿热偏盛，建议多吃清淡食物，如苦瓜、绿豆汤，避免油炸辛辣。", "每天散步30分钟，有助排湿。"
        )
    }
    dx_components = []
    treatment_components = []
    if color in color_map:
        dx_components.append(color_map[color][0])
        treatment_components.append(f"【饮食建议】{color_map[color][1]}")
        treatment_components.append(f"【运动建议】{color_map[color][2]}")
    else:
        dx_components.append("舌色未见明显异常")
    if coating in coating_map:
        dx_components.append(coating_map[coating][0])
        treatment_components.append(f"【饮食建议】{coating_map[coating][1]}")
        treatment_components.append(f"【运动建议】{coating_map[coating][2]}")
    else:
        dx_components.append("舌苔未见明显异常")
    diagnosis = "综合辨证：" + "，".join(dx_components)
    treatment = "\n".join(treatment_components)
    if cancer != "正常舌头" and cancer != 0:
        diagnosis += "（可能伴局部病变）"
        treatment += "\n【健康提示】：检测到舌部异常，建议尽快就医检查。"
    treatment += "\n【扩展诊断】：本诊断结合中医理论与现代检测技术，建议患者结合临床症状，必要时进一步检查。"
    return diagnosis, treatment


import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import base64
import cv2
import base64

def generate_html_report(image_path, color, coating, cancer, boxes, diagnosis, treatment,
                         output_html="tongue_report.html"):
    """
    生成包含检测框标注的 HTML 报告。
    boxes 字典包含：
        - 'tongue' : 舌头检测框 (x1, y1, x2, y2)
    """
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法加载图像用于生成报告。")
        return
    if boxes.get('tongue') is not None:
        x1, y1, x2, y2 = boxes['tongue']
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)


    # 将 OpenCV 图像转换为 Pillow 图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # 设置字体（确保 simsun.ttc 文件在当前目录下）
    font_path = "simsun.ttc"
    try:
        font = ImageFont.truetype(font_path, 40, encoding="utf-8")
    except IOError:
        print(f"错误：无法加载字体文件 {font_path}。请确保字体文件存在于当前目录。")
        return


    # 添加 color、coating 和 cancer 预测结果文字标注
    text_position = (10, 30)  # 文字标注的位置，初始位置可以调整

    # 标注舌色预测
    draw.text(text_position, f"舌色: {color}", font=font, fill=(0, 255, 0))
    text_position = (10, 70)  # 文字标注的位置向下偏移

    # 标注舌苔预测
    draw.text(text_position, f"舌苔: {coating}", font=font, fill=(0, 0, 255))
    text_position = (10, 110)  # 文字标注的位置继续向下偏移

    # 标注舌病变预测
    draw.text(text_position, f"病变: {cancer}", font=font, fill=(255, 255, 0))

    # 将 Pillow 图像转换回 OpenCV 图像
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 保存已标注的图像
    annotated_image_path = "annotated_report.png"
    cv2.imwrite(annotated_image_path, image)

    # 将图像转换为 base64 编码
    with open(annotated_image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    # 生成 HTML 内容
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>舌诊综合报告</title>
        <style>
            body {{
                font-size: 18px; /* 设置全局字体大小 */
            }}
            h1 {{
                font-size: 24px; /* 设置主标题字体大小 */
            }}
            h2 {{
                font-size: 22px; /* 设置副标题字体大小 */
            }}
            ul {{
                font-size: 20px; /* 设置列表字体大小 */
            }}
            p {{
                font-size: 20px; /* 设置段落字体大小 */
            }}
        </style>
    </head>
    <body>
        <h1>舌诊综合报告</h1>
        <h2>预测结果</h2>
        <ul>
            <li>舌色预测结果：{color}</li>
            <li>舌苔预测结果：{coating}</li>
            <li>舌病变预测结果：{cancer}</li>
        </ul>
        <h2>辨证与调理建议</h2>
        <p>{diagnosis}</p>
        <p>{treatment}</p>
        <h2>检测图像</h2>
        <img src="data:image/png;base64,{encoded_string}" alt="Annotated Tongue Image" style="max-width:100%;height:auto;"/>
    </body>
    </html>
    """
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML 报告已生成：{output_html}")
    return annotated_image_path

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    test_image = r"img_11.png"  # 替换为实际图片路径
    color_report,coating_report,cancer_report,tongue_annotated,diagnosis,treatment=tongue_diagnose_sum(test_image)
    print(color_report,coating_report,cancer_report,tongue_annotated,diagnosis,treatment)