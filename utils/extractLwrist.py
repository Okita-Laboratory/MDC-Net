import numpy as np
import os
bodyName = ['hip', 'r_hip', 'r_knee', 'r_foot', 'l_hip', 'l_knee', 'l_foot', 'spine', 'thorax',
            'neck', 'head', 'l_shoulder', 'l_elbow', 'l_wrist', 'r_shoulder', 'r_elbow', 'r_wrist']
# 仅输出的部位名称
selected_body_parts = ['l_wrist']  # 指定输出的部位，可以根据需要修改

data = np.loadtxt('/home/ghy/workspace/MOCAP/output/60min/last30min/last30min/h36m_keypoints.txt')
total_frame = data.shape[0] // 17
data = data.reshape((total_frame, 17, 3))
timeStamp = np.arange(total_frame).reshape(-1, 1) * 0.02
print(data.shape)
# 找到所选部位在 bodyName 中的索引
selected_indices = [bodyName.index(part) for part in selected_body_parts if part in bodyName]

# 输出指定部位的数据
for index in selected_indices:
    saveData = np.concatenate((timeStamp, data[:, index, :]), axis=1)
    dir_txt = os.path.join('/home/ghy/workspace/MOCAP/output/60min/last30min/last30min/Lwrist.txt')
    os.makedirs(os.path.dirname(dir_txt), exist_ok=True)
    np.savetxt(dir_txt, saveData, fmt='%.2f ' + '%.6f ' * (saveData.shape[1] - 1))

print(f'已保存选定部位的数据：{selected_body_parts}')
