from PIL import Image
import os
from pathlib import Path

def process_single_image(input_path, output_path, target_size=(3384, 384), rotation_angle=0):
    """处理单张图片：旋转并调整尺寸"""
    # 打开图片
    with Image.open(input_path) as img:
        # 先旋转图片（如果需要）
        if rotation_angle != 0:
            # 使用expand=True确保旋转后的图片完整显示
            rotated_img = img.rotate(rotation_angle, expand=True)
        else:
            rotated_img = img
        
        # 调整图片尺寸
        resized_img = rotated_img.resize(target_size, Image.LANCZOS)
        
        # 保存图片，quality参数用于压缩，范围1-95，95为最高质量
        resized_img.save(output_path, quality=95)
        
        return rotated_img.size, resized_img.size

def process_all_images(input_folder, output_folder, target_size=(3384, 384), rotation_angle=0):
    """处理文件夹下的所有图片"""
    # 创建输出文件夹（如果不存在）
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    
    # 获取所有图片文件
    image_files = []
    for fmt in supported_formats:
        image_files.extend(Path(input_folder).glob(f'*{fmt}'))
        image_files.extend(Path(input_folder).glob(f'*{fmt.upper()}'))
    
    if not image_files:
        print(f"在 {input_folder} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始处理...")
    
    # 处理每张图片
    for i, img_path in enumerate(image_files, 1):
        try:
            # 构建输出路径，保持原文件名
            output_path = Path(output_folder) / img_path.name
            
            # 处理图片
            original_size, final_size = process_single_image(
                str(img_path), 
                str(output_path), 
                target_size, 
                rotation_angle
            )
            
            print(f"[{i}/{len(image_files)}] 处理完成: {img_path.name}")
            print(f"  原始尺寸: {img_path.stat().st_size // 1024}KB")
            print(f"  旋转后尺寸: {original_size}")
            print(f"  最终尺寸: {final_size}")
            
        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {str(e)}")
            continue
    
    print(f"\n所有图片处理完成！输出保存在: {output_folder}")

if __name__ == "__main__":
    # 输入输出路径
    input_folder = "/mnt/data/user/zhao_jun/sfg/DIP/new/Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights/realtest"
    output_folder = "/mnt/data/user/zhao_jun/sfg/DIP/new/Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights/realtest_input"
    
    # 设置参数
    target_size = (512,512)  # 目标尺寸
    rotation_angle = 90  # 旋转角度（度）：正值为逆时针，负值为顺时针，0为不旋转
    
    # 处理所有图片
    process_all_images(input_folder, output_folder, target_size, rotation_angle)
    
    # 测试查看原始图片尺寸
    print("\n测试图片尺寸:")
    test_img = Image.open("/mnt/data/user/zhao_jun/sfg/DIP/Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights/dataset/Flare7Kpp/Flare-R/Compound_Flare/000000.png")
    print(f"测试图片尺寸: {test_img.size}")  # 输出 (宽, 高)

