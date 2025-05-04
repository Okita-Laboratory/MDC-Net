from torch import tensor
from utils import *
from config import Config, update_config
from utils.visualization import render_animation
import argparse
from utils.script import *
from utils.training import Trainer
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def my_pose_generator(data_set, model_select, diffusion, cfg, mode=None, action=None, nrow=1):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    print("22222")
    model_select = model_select
    action = action
    traj_np = None
    j = None
    while True:
        poses = {}
        draw_order_indicator = -1
        for k in range(0, nrow):
            data = data_set.sample()
            # data.shape (1, 125, 17, 3)
            gt = data[0].copy()
            # gt gt.shape (125, 17, 3)

            # remove first sk
            gt[:, :1, :] = 0
            data[:, :, :1, :] = 0

            if draw_order_indicator == -1:
                poses['context'] = gt
                poses['gt'] = gt
            else:
                poses[f'HumanMAC_{draw_order_indicator + 1}'] = gt
                poses[f'HumanMAC_{draw_order_indicator + 2}'] = gt

            gt = np.expand_dims(gt, axis=0)
            # gt.shape (1, 125, 17, 3)

            # remove first sk
            traj_np = gt[..., 1:, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred + cfg.t_fut, -1])
            # traj_np.shape (1, 125, 48)

            traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)

            mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, cfg, mode=mode)
            # mode_dict['mask'].shape ([5, 125, 48])

            sampled_motion = diffusion.sample_ddim(model_select,
                                                   traj_dct,
                                                   traj_dct_mod,
                                                   mode_dict)

            traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
            traj_est = traj_est.cpu().numpy()
            traj_est = post_process(traj_est, cfg)

            if k == 0:
                for j in range(traj_est.shape[0]):
                    poses[f'HumanMAC_{j}'] = traj_est[j]
            else:
                for j in range(traj_est.shape[0]):
                    poses[f'HumanMAC_{j + draw_order_indicator + 2 + 1}'] = traj_est[j]

            if draw_order_indicator == -1:
                draw_order_indicator = j
            else:
                draw_order_indicator = j + draw_order_indicator + 2 + 1

        yield poses
def animation():
    # 创建图形和轴
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')

    # 初始化函数：图形的背景
    def init():
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1, 1)
        return ln,

    # 更新函数：这里定义了如何更新每一帧
    def update(frame):
        xdata.append(frame)
        ydata.append(np.sin(frame))
        ln.set_data(xdata, ydata)
        return ln,

    # 创建FuncAnimation实例
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128),
                        init_func=init, blit=True)

    plt.show()

def after_train_step():

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))

    dataset, dataset_multi_test = dataset_split(cfg)
    model, diffusion = create_model_and_diffusion(cfg)


    # after_train_step
    pose_gen = my_pose_generator(data_set=dataset['train'], model_select=model, diffusion=diffusion, cfg=cfg, mode='gif')
    all_poses = next(pose_gen)
    # for key, value in all_poses.items():
    #     print(f'{key}: {value}, value.shape={value.shape}')
    #all_poses['gt'], all_poses['HumanMAC_0'] = all_poses['HumanMAC_0'], all_poses['gt']
    render_animation(dataset['train'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, cfg.t_pred, cfg.t_fut, ncol=4,
                      output=os.path.join(cfg.gif_dir, f'training_{iter}.gif'))

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        default='h36m', help='h36m or humaneva')
    parser.add_argument('--mode', default='train', help='train / eval / pred / switch/ control/ zero_shot')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    parser.add_argument('--milestone', type=list, default=[75, 150, 225, 275, 350, 450])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_model_interval', type=int, default=10)
    parser.add_argument('--save_gif_interval', type=int, default=10)
    parser.add_argument('--save_metrics_interval', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/h36m_ckpt.pt')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_switch_num', type=int, default=10)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=3)
    args = parser.parse_args()
    #animation()
    after_train_step()