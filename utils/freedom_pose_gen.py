import os
import numpy as np
from utils.pose_gen import pose_generator
import torch
import copy
# --coding:utf-8--
def freedom_pose_gen(cfg, model, diffusion, dataset, action_path):


    print("freedom mode start・・・・・・")
    length = len(action_path)
    print(f'There are {length} actions: {action_path}')
    final_pose = np.empty((0, 17, 3))
    for i in range(length-1):
        pose_gen = pose_generator(data_set=dataset['test'],
                                  model_select=model,
                                  diffusion=diffusion,
                                  cfg=cfg,
                                  mode='freedom',
                                  action=action_path[i:i+2],
                                  nrow=cfg.vis_row)
        all_poses = next(pose_gen)
        temp1 = all_poses['completion'][:, cfg.t_his: (cfg.t_his + cfg.t_pred), :, :]
        temp1 = np.squeeze(temp1, axis=0)

        data0 = np.loadtxt(action_path[i])
        num0_frames = data0.shape[0] // 17
        traj_his = data0.reshape((num0_frames, 17, 3))

        data1 = np.loadtxt(action_path[i+1])
        num1_frames = data1.shape[0] // 17
        traj_fut = data1.reshape((num1_frames, 17, 3))

        temp = np.concatenate((traj_his, temp1, traj_fut), 0)
        print(f'traj_his:{traj_his.shape}, temp1:{temp1.shape}, traj_fut:{traj_fut.shape}')

        if i >= 1:
            final_pose = np.concatenate((final_pose, temp[num0_frames:, :, :]), axis=0)
        else:
            final_pose = np.concatenate((final_pose, temp), axis=0)

    # final_pose 保存为 txt 文件
    print('total frame=', final_pose.shape)

    temp_copy = copy.deepcopy(final_pose)
    # temp_copy[:, :, [2]] = - temp_copy[:, :, [2]]
    # temp_copy[:, :, [0, 1, 2]] = temp_copy[:, :, [0, 2, 1]]

    final_pose_output_path = '/home/ghy/workspace/MOCAP/output/reviewer1/h36m_keypoints.txt'
    output_dir = os.path.dirname(final_pose_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(final_pose_output_path, temp_copy.reshape(-1, 3), fmt='%.6f')
    print(f"已保存到: {output_dir}")

def sample_one_action(cfg, dataset, action, sample_mode, start, end):
    sample = dataset['test'].my_sample_action(dataset_type=cfg.dataset, action_category=action, sample_mode=sample_mode, start=start, end=end)
    print('total frame=', sample.shape)
    sample[:, :, 0, :] = 0
    final_pose_output_path = f'/home/ghy/workspace/MOCAP/output/sample/{action}/h36m_keypoints.txt'
    output_dir = os.path.dirname(final_pose_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(final_pose_output_path, sample.reshape(-1, 3), fmt='%.6f')
    print(f"已保存到: {output_dir}")