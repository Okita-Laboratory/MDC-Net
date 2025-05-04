import numpy as np
import scipy.io as sio
import xml.etree.ElementTree as ET

def mat_to_xml(mat_file, xml_file):
    """
    将 .mat 文件内容保存为 XML 文件
    """
    # 读取 .mat 文件
    mat_data = sio.loadmat(mat_file)
    print(mat_data)

    # 设置 NumPy 打印选项，确保打印完整数据
    np.set_printoptions(threshold=np.inf)

    # 创建 XML 根节点
    root = ET.Element("MatData")

    for key, value in mat_data.items():
        if key.startswith("__"):  # 跳过 MATLAB 内置变量
            continue

        variable = ET.SubElement(root, "Variable", name=key)

        if isinstance(value, (int, float, str)):  # 简单变量
            variable.text = str(value)
        elif isinstance(value, (list, tuple)):  # 列表或元组
            for item in value:
                ET.SubElement(variable, "Item").text = str(item)
        elif hasattr(value, "shape"):  # 多维数组
            # 确保完整保存每行数据
            for i, row in enumerate(value):
                row_elem = ET.SubElement(variable, "Row", index=str(i))
                row_elem.text = " ".join(map(str, row.flatten()))
        else:
            variable.text = "Unsupported data type"

    # 保存为 XML 文件
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"保存 XML 文件成功: {xml_file}")

    # 恢复默认的打印设置
    np.set_printoptions(threshold=1000)



# 示例使用
mat_file_path = "/home/ghy/workspace/dataset/BMLMovi/AMASS/F_amass_Subject_2.mat"  # 替换为你的 .mat 文件路径
xml_file_path = "/home/ghy/workspace/dataset/BMLMovi/F_amass_Subject_2.xml"  # 替换为输出的 .xml 文件路径
mat_to_xml(mat_file_path, xml_file_path)
