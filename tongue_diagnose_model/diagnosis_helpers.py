from tongue_diagnose_model.sum_predict_second import DIAGNOSIS_RULES, sum_predict
import os

def get_diagnosis_report(image_path):
    """
    辅助函数：获取舌诊报告 - 演示模式
    
    Args:
        image_path: 图像路径
    
    Returns:
        tuple: 包含所有报告信息的元组
    """
    # 演示模式 - 固定返回一个完整的诊断报告样例
    
    # 固定的舌诊特征结果
    demo_features = {
        "舌色": "淡红舌",
        "舌形": "齿痕舌",
        "苔色": "薄白苔",
        "苔质": "薄苔",
        "舌态": "正常舌态",
        "舌神": "荣舌",
        "舌脉": "舌脉曲张"
    }
    
    # 基础特征分析文本
    color_report = f"舌色类型: {demo_features['舌色']}\n{DIAGNOSIS_RULES['舌色'][demo_features['舌色']]['分析']}"
    coating_color_report = f"苔色类型: {demo_features['苔色']}\n{DIAGNOSIS_RULES['苔色'][demo_features['苔色']]['分析'] if demo_features['苔色'] in DIAGNOSIS_RULES['苔色'] else '显示胃气充盈，气血正常。'}"
    shape_report = f"舌形类型: {demo_features['舌形']}\n{DIAGNOSIS_RULES['舌形'][demo_features['舌形']]['分析']}"
    coating_texture_report = f"苔质类型: {demo_features['苔质']}\n{DIAGNOSIS_RULES['苔质'][demo_features['苔质']]['分析']}"
    
    # 异常检测报告
    cancer_report = "未发现明显异常"
    
    # 诊断与治疗建议
    diagnosis = f"""【基础特征分析】
舌色: {demo_features['舌色']}
舌形: {demo_features['舌形']}
苔色: {demo_features['苔色']}
苔质: {demo_features['苔质']}
舌态: {demo_features['舌态']}
舌神: {demo_features['舌神']}
舌脉: {demo_features['舌脉']}

【中医辨证】
舌色 → 正常舌色: 气血调和，脏腑功能正常。
舌形 → 脾虚证: 脾气虚弱，水湿内停，常见于消化不良。
苔色 → 正常苔色: 显示胃气充盈、津液充足。
苔质 → 正常苔质: 胃气充盈，津液未伤，正常生理现象。
舌脉 → 血瘀阻滞: 脉络曲张，提示血液循环不畅。
舌神 → 正气未衰: 舌色红活润泽，活动自如，反映机体正气充足，预后良好。

复合证型: 脾虚湿盛"""

    treatment = """推荐方剂: 参苓白术散

调理建议:
1. 饮食宜清淡，易消化，避免油腻、生冷、辛辣刺激性食物
2. 注意保暖，避免受凉，保持良好作息
3. 适当活动，促进气血运行
4. 保持情绪稳定，避免过度忧思

中药治疗原则:
* 健脾益气，化湿利水
* 活血化瘀，通络止痛

调理期望:
坚持调理2-3个月，可望明显改善脾胃功能，减轻水湿内停症状"""
    
    # 返回完整的报告
    return color_report, coating_color_report, shape_report, coating_texture_report, cancer_report, diagnosis, treatment 