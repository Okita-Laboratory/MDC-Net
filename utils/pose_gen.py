import numpy as np
import torch
from torch import tensor
from utils import *
from utils.script import sample_preprocessing


def pose_generator(data_set, model_select, diffusion, cfg, mode=None,
                   action=None, nrow=1):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    traj_np = None
    j = None
    while True:
        poses = {}
        draw_order_indicator = -1
        if mode == 'joint':
            data0 = data_set.my_sample_iter_action(action[0], cfg.dataset, sample_mode='last') # sample the history motion
            data1 = data_set.my_sample_iter_action(action[1], cfg.dataset, sample_mode='first') # sample the future motion
            # get the last t_his frames of history motion
            temp0 = data0[:, -cfg.t_his:, :, :]
            #
            traj_m = np.array(torch.tensor(temp0[:, -1, :, :]).repeat(1, cfg.t_pred, 1, 1))
            # get the fisrt t_fyt frames of future motion
            temp1 = data1[:, 0:cfg.t_fut, :, :]
            data = np.concatenate((temp0, traj_m, temp1), 1)
        elif mode == 'freedom':

            data0 = np.loadtxt(action[0])
            num_frames = data0.shape[0] // 17
            data0 = data0.reshape((num_frames, 17, 3))
            data0 = np.expand_dims(data0, axis=0)

            data1 = np.loadtxt(action[1])
            num_frames = data1.shape[0] // 17
            data1 = data1.reshape((num_frames, 17, 3))
            data1 = np.expand_dims(data1, axis=0)

            # get the last t_his frames of history motion
            temp0 = data0[:, -cfg.t_his:, :, :]
            #
            traj_m = np.array(torch.tensor(temp0[:, -1, :, :]).repeat(1, cfg.t_pred, 1, 1))
            # get the fisrt t_fyt frames of future motion
            temp1 = data1[:, 0:cfg.t_fut, :, :]
            data = np.concatenate((temp0, traj_m, temp1), 1)
        else:
            raise NotImplementedError(f"unknown pose generator mode: {mode}")

        # gt
        gt = data[0].copy()
        gt[:, :1, :] = 0
        data[:, :, :1, :] = 0

        if mode == 'joint':
            poses = {}
            traj_np = data[..., 1:, :].reshape([data.shape[0], cfg.t_his + cfg.t_pred + cfg.t_fut, -1])
        if mode == 'freedom':
            poses = {}
            traj_np = data[..., 1:, :].reshape([data.shape[0], cfg.t_his + cfg.t_pred + cfg.t_fut, -1])
        if mode == 'switch':
            poses = {}
            traj_np = data[..., 1:, :].reshape([data.shape[0], cfg.t_his + cfg.t_pred + cfg.t_fut, -1])
        elif mode == 'pred' or mode == 'gif' or 'fix' in mode or mode == 'zero_shot':
            if draw_order_indicator == -1:
                poses['context'] = gt
                poses['gt'] = gt
            else:
                poses[f'HumanMAC_{draw_order_indicator + 1}'] = gt
                poses[f'HumanMAC_{draw_order_indicator + 2}'] = gt
            gt = np.expand_dims(gt, axis=0)
            traj_np = gt[..., 1:, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred + cfg.t_fut, -1])

        traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)
        mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, cfg, mode=mode)
        sampled_motion = diffusion.sample_ddim(model_select,
                                               traj_dct,
                                               traj_dct_mod,
                                               mode_dict)
        # traj_est is the generated motion
        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.cpu().numpy()
        traj_est = post_process(traj_est, cfg)
        if mode == 'joint':
            data0 = np.delete(data0, 0, axis=2) # 删除hip这个关节
            data1 = np.delete(data1, 0, axis=2)
            print('data0.shape:', data0.shape)
            print('data1.shape:', data1.shape)
            data0 = post_process(data0, cfg)

            data1 = post_process(data1, cfg)
            """
            data0 history; data1 future; traj_est prediction
            """
            traj_est = np.concatenate([data0, data1, traj_est])
        if mode == 'freedom':
            traj_est = traj_est
            traj_est = np.squeeze(traj_est, axis=0)

        poses[f'completion'] = traj_est[j]

        yield poses
