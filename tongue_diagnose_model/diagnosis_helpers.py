from datetime import datetime

def get_diagnosis_report(features, diagnosis_text, treatment_text, image_path=None):
    """
    将舌诊分析结果格式化为诊断报告
    
    参数:
        features: 舌象特征字典
        diagnosis_text: 诊断文本
        treatment_text: 治疗建议文本
        image_path: 舌象图像路径
    
    返回:
        格式化的HTML报告和纯文本报告
    """
    # 预处理诊断和治疗文本（修复换行符）
    diagnosis_processed = diagnosis_text.replace('\n', '<br>')
    treatment_processed = treatment_text.replace('\n', '<br>')
    
    # 准备图像嵌入HTML（如果提供了图像路径）
    img_src = ""
    if image_path:
        try:
            import base64
            from pathlib import Path
            with open(image_path, "rb") as img_file:
                img_format = Path(image_path).suffix[1:]  # 获取扩展名，去掉点
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                img_src = f"data:image/{img_format};base64,{img_data}"
        except Exception as e:
            print(f"图像嵌入错误: {e}")
    
    # 生成HTML格式报告
    html_report = f"""
    <div class="report-section tongue-section">
        <h3>舌诊分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
        <div class="tongue-content">
            <div class="tongue-image">
                {"<img src='" + img_src + "' alt='舌象分析图'>" if img_src else ""}
            </div>
            <div class="diagnosis-text">
                <div class="diagnosis-summary">
                    <p>{diagnosis_processed}</p>
                    <h4>治疗建议</h4>
                    <p>{treatment_processed}</p>
                </div>
            </div>
        </div>
    </div>
    <hr>
    """
    
    # 生成纯文本格式报告（用于聊天历史）
    text_report = f"""
初步诊断结论: {diagnosis_text}
治疗建议: {treatment_text}
    """
    
    return html_report, text_report.strip() 