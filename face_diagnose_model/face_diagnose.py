import cv2
from .picseg import pic_seg
from .color_distance import skin_color_detection, description as face_color_description
import os

FACE_DIAGNOSE_DETAIL = {
    "庭": {
        "diagnosis": "前额区域诊断",
        "analysis_map": {
            "红": "心气充沛、心血充盈",
            "白": "近期劳累、休息不足",
            "黑": "心肾不交早期表现",
            "黄": "心脾功能不协调"
        }
    },
    "左颊": {
        "diagnosis": "肝脏区域诊断",
        "analysis_map": {
            "红": "肝血充足，肝气条达",
            "白": "用眼过度或情绪焦虑",
            "黑": "熬夜影响排毒功能",
            "黄": "情绪波动影响气血"
        }
    },
    "右颊": {
        "diagnosis": "肺部区域诊断", 
        "analysis_map": {
            "红": "肺气充足功能正常",
            "白": "肺卫功能稍弱",
            "黑": "环境因素影响肺部",
            "黄": "水液代谢异常"
        }
    },
    "鼻": {
        "diagnosis": "脾胃区域诊断",
        "analysis_map": {
            "红": "脾胃运化正常",
            "白": "脾胃功能稍弱",
            "黑": "脾胃受寒阳气损",
            "黄": "功能暂时性异常"
        }
    },
    "颌": {
        "diagnosis": "肾脏区域诊断",
        "analysis_map": {
            "红": "肾精充足阴阳衡",
            "白": "肾阳稍不足",
            "黑": "过劳影响肾功能", 
            "黄": "水液调节调整中"
        }
    }
}

def face_diagnose_sum(image_path,save_dir):
    """
    面部诊断集成函数
    返回：(diagnosis_report, treatment_advice, annotated_img_path)
    """
    try:
        # 确保存储目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 修改为绝对路径或相对于当前文件的路径
        faceseg_dir = os.path.join(os.path.dirname(__file__), 'faceseg', 'roi_images')
        os.makedirs(faceseg_dir, exist_ok=True)
        
        # 步骤1：面部区域分割
        pic_seg(image_path,faceseg_dir)
        
        # 步骤2：遍历分析各区域
        analysis_results = []
        
        for filename in os.listdir(faceseg_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                region_key = os.path.splitext(filename)[0]
                # 从region_key中提取实际区域名称（去掉_roi后缀）
                key_name = region_key.replace('_roi', '')
                region_name = {
                    'jia_left': '左颊', 
                    'jia_right': '右颊',
                    'ke': '颌',
                    'ming_tang': '鼻',
                    'ting': '庭'
                }.get(key_name, '未知区域')
                
                # 颜色分析
                img_path = os.path.join(faceseg_dir, filename)
                color_analysis_dir = os.path.join(faceseg_dir,"color_anlysis_results")
                # 确保目录存在
                if not os.path.exists(color_analysis_dir):
                    os.makedirs(color_analysis_dir)
                color = skin_color_detection(img_path, region_name, color_analysis_dir)
                
                # 获取区域对应的器官信息和详细描述
                try:
                    # 跳过未知区域
                    if region_name == '未知区域':
                        continue
                        
                    region_detail = FACE_DIAGNOSE_DETAIL.get(region_name, {})
                    organ_info = region_detail.get("diagnosis", "未知区域")
                    detailed_description = face_color_description(region_name, color)
                    
                    # 按照新格式输出，改为"您的【区域名】偏 颜色"格式
                    formatted_result = f"您的【{region_name}】偏 {color}\n{organ_info}：{detailed_description}"
                    analysis_results.append(formatted_result)
                except Exception as e:
                    print(f"分析错误: {e}")
                    # 回退到简单格式
                    if region_name != '未知区域':
                        region_detail = FACE_DIAGNOSE_DETAIL.get(region_name, {})
                        analysis = region_detail.get("analysis_map", {}).get(color, "无明显异常")
                        analysis_results.append(f"您的【{region_name}】偏 {color}\n{region_detail.get('diagnosis', '')}：{analysis}")
        
        # 步骤3：生成综合报告，去掉标题
        diagnosis_report = "\n\n".join(analysis_results)
        
        # # 步骤4：生成治疗建议
        # treatment_advice = "根据面诊分析，建议：\n1. 调整作息，保证充足睡眠\n2. 注意饮食均衡，增加蔬果摄入\n3. 适当运动，增强体质"
        
        # 使用已生成的标注图像
        annotated_img_path = os.path.join(faceseg_dir,"annotated_image.jpg")
        
        return diagnosis_report,  annotated_img_path
        
    except Exception as e:
        print(f"面诊过程中出现异常: {e}")
        return "诊断失败", None

def generate_treatment_advice(analysis_results):
    """生成综合调理建议（增强版）"""
    advice = []
    
    # 根据区域添加建议
    if any("心气" in res for res in analysis_results):
        advice.append("🫀 养心建议：保持规律作息，可适量食用红枣、龙眼等补心食物")
    if any("肝血" in res for res in analysis_results):
        advice.append("🌿 护肝建议：避免熬夜，保持心情舒畅，可饮用菊花枸杞茶")
    if any("肺气" in res for res in analysis_results):
        advice.append("🌬️ 润肺建议：适当进行深呼吸锻炼，保持空气流通")
    if any("肾精" in res for res in analysis_results):
        advice.append("💧 补肾建议：适量食用黑色食物如黑芝麻、黑豆，避免过度劳累")
    
    # 根据颜色添加建议
    if any("红" in res and "心气" in res for res in analysis_results):
        advice.append("❄️ 清热建议：可适量饮用莲子心茶以降心火")
    if any("白" in res and "脾胃" in res for res in analysis_results):
        advice.append("🍚 健脾建议：增加山药、小米等易消化食物摄入")
    
    return "\n".join(advice) if advice else "当前状态良好，保持健康生活方式即可"

def annotate_facial_regions(image_path):
    """生成带标注的面部图像（示例实现）"""
    img = cv2.imread(image_path)
    # 实际应添加面部区域标注逻辑
    output_path = "annotated_face.jpg"
    cv2.imwrite(output_path, img)
    return output_path

if __name__ == '__main__':
    diagnosis, advice, img_path = face_diagnose_sum("four_color_face_sample/black.png")