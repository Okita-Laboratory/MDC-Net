import os
from PIL import Image, ImageDraw, ImageFont
import shutil
def extractGif(name):
    # 打开 GIF 文件
    gif_path = '/home/ghy/workspace/MY/extractGif/gif/' + name + '.gif'
    gif = Image.open(gif_path)

    # 初始化帧计数器
    frame_number = 0

    while True:
        # 保存当前帧为 PNG 文件
        frame_path = f'/home/ghy/workspace/MY/extractGif/makeimge/{name}/{frame_number}.png'
        if not os.path.exists(f'/home/ghy/workspace/MY/extractGif/makeimge/{name}'):
            os.makedirs(f'/home/ghy/workspace/MY/extractGif/makeimge/{name}')
        gif.save(frame_path, 'PNG')

        # 移动到下一个帧
        frame_number += 1
        try:
            gif.seek(gif.tell() + 1)
        except EOFError:
            # 到达最后一帧，退出循环
            break


def get_image_files_from_folder(folder, num_list):
    # 获取文件夹中所有图片文件的路径
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(image_files)
    # 过滤出文件名包含在num_list中的文件
    filtered_files = [f for f in image_files if os.path.basename(f) in num_list]

    # 按数字排序文件
    sorted_files = sorted(filtered_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

    return sorted_files

def concatenate_images_horizontally(image_files):
    images = [Image.open(img_file) for img_file in image_files]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights) + 50  # 50 pixels for the text height

    new_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    draw = ImageDraw.Draw(new_image)

    # 使用指定的字体文件和字体大小
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # 替换为你系统中存在的字体文件路径
    font_size = 20  # 调整字体大小
    font = ImageFont.truetype(font_path, font_size)

    x_offset = 0
    for img, img_file in zip(images, image_files):
        file_name = os.path.basename(img_file)
        file_name = file_name[:file_name.rfind('.')]  # 去掉文件后缀
        text = f"Frame {file_name.split()[-1]}"

        # 计算文本宽度并居中对齐
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_offset + (img.width - text_width) // 2
        draw.text((text_x, 5), text, fill="black", font=font)

        new_image.paste(img, (x_offset, 50))
        x_offset += img.width

    return new_image


def imagepro(name, numlist):
    # 替换成包含图片文件的文件夹路径
    folder = '/home/ghy/workspace/MY/extractGif/makeimge/' + name
    image_files = get_image_files_from_folder(folder, numlist)
    if image_files:
        result_image = concatenate_images_horizontally(image_files)
        result_image.show()  # 显示合成后的图片
        result_image.save('/home/ghy/workspace/MY/extractGif/makeimge/' + name + '/combined_image.jpg')  # 保存合成后的图片
    else:
        print("No image files found in the folder.")


def copy_specific_files(src_folder, dst_folder, keyword):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for file_name in os.listdir(src_folder):
        if keyword + '.png' == file_name:
            src_file = os.path.join(src_folder, file_name)
            dst_file = os.path.join(dst_folder, file_name)
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")

if __name__=="__main__":
    motion = ['Greeting_joint_Phoning', 'SittingDown_joint_Walking', 'Sitting_joint_Greeting',
              'Walking_joint_Sitting', 'Walking_joint_SittingDown', 'Walking_joint_WalkDog']
    Numlist = ['13.png', '14.png', '140.png', '155.png', '170.png', '185.png', '200.png', '215.png', '230.png', '232.png', '242.png']
    for name in motion:
        extractGif(name)
        print(f"已提取{name}")
        # for num in Num:
        #     src_folder = '/home/ghy/workspace/MY/extractGif/'
        #     dst_folder = '/home/ghy/workspace/MY/extractGif/makeimge/' + name + '/'
        #     copy_specific_files(src_folder, dst_folder, num)
        print(f"已转移{name}")
        imagepro(name, Numlist)
        print(f"已制作{name}合成图")

