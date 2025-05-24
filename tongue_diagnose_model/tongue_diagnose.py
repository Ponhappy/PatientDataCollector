import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from .sum_predict_second import TongueAnalysisSystem, DiagnosisEngine, ReportGenerator

def tongue_diagnose_sum(image_path, user_dir=None, use_model=True):
    """
    执行舌诊分析并生成诊断报告
    
    参数:
        image_path: 舌头图像路径
        user_dir: 用户目录，用于保存结果
        use_model: 是否使用模型进行预测
    
    返回:
        (html_report, text_report, image_path)
    """
    # 保存结果的目录
    results_dir = os.path.join(user_dir, "tongue_results") if user_dir else "tongue_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 带时间戳的输出文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_html = os.path.join(results_dir, f"tongue_report_{timestamp}.html")
    
    # 检查输入图像是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入图像不存在: {image_path}")
    
    # 初始化分析系统
    system = TongueAnalysisSystem()
    
    # 提取特征
    features = {
        "舌色": system.predict_category(image_path, "舌色"),
        "舌形": system.predict_category(image_path, "舌形"),
        "苔色": system.predict_category(image_path, "苔色"),
        "苔质": system.predict_category(image_path, "苔质"),
        "舌态": system.predict_category(image_path, "舌态"),
        "舌神": system.predict_category(image_path, "舌神"),
        "舌脉": system.predict_category(image_path, "舌脉")
    }
    
    # 检测舌头区域 (仅用于完整报告)
    _, tongue_box = system.detect_tongue(image_path)
    boxes = {'tongue': tongue_box.tolist()} if tongue_box is not None else None
    
    # 生成诊断
    diagnosis_text, treatment_text = DiagnosisEngine.analyze(features)
    
    # 处理诊断文本 - 移除重复的基础特征部分
    clean_diagnosis = diagnosis_text
    if "【基础特征分析】" in diagnosis_text and "【中医辨证】" in diagnosis_text:
        try:
            # 只保留中医辨证部分
            clean_diagnosis = diagnosis_text.split("【中医辨证】")[1].strip()
            clean_diagnosis = "【中医辨证】" + clean_diagnosis
        except:
            # 如果分割失败，保持原文本
            clean_diagnosis = diagnosis_text
    
    # 检查治疗建议是否为空
    if not treatment_text or treatment_text.strip() == "":
        treatment_text = """
1. 请根据具体症状向专业医师咨询
2. 合理饮食，避免辛辣刺激食物
3. 保持良好作息，避免过度劳累
        """
    
    # 生成UI友好的HTML报告
    # 将图像转换为base64格式以嵌入HTML
    try:
        import base64
        from pathlib import Path
        with open(image_path, "rb") as img_file:
            img_format = Path(image_path).suffix[1:]  # 获取扩展名，去掉点
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            img_src = f"data:image/{img_format};base64,{img_data}"
    except Exception as e:
        print(f"图像嵌入错误: {e}")
        img_src = ""
    
    # 预处理诊断文本
    diagnosis_processed = clean_diagnosis.replace('\n', '<br>')
    treatment_processed = treatment_text.replace('\n', '<br>')
    
    # 生成为UI优化的HTML报告
    html_report = f"""
    <div class="report-section tongue-section">
        <h3>舌诊分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
        <div class="tongue-content" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div class="tongue-image">
                <img src="{img_src}" alt="舌象分析图" style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px;">
            </div>
            <div class="diagnosis-text">
                <p><strong>基础特征:</strong></p>
                <ul>
                    {"".join(f"<li><b>{k}</b>: {v}</li>" for k, v in features.items() if k != "tongue_box")}
                </ul>
                <p><strong>诊断结论:</strong></p>
                <p>{diagnosis_processed}</p>
                <p><strong>治疗建议:</strong></p>
                <p>{treatment_processed}</p>
            </div>
        </div>
    </div>
    <hr>
    """
    
    # 同时使用ReportGenerator生成完整的HTML报告文件（用于导出）
    ReportGenerator.generate_html_report(
        image_path, features, diagnosis_text, treatment_text, 
        boxes=boxes, output_html=output_html
    )
    
    # 为聊天历史生成纯文本报告
    text_report = f"""
初步诊断结论: {clean_diagnosis}
治疗建议: {treatment_text}
    """
    
    # 返回报告和原始图像路径
    return html_report, text_report.strip(), image_path 