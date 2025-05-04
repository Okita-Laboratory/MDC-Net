import os
from utils.pose_gen import pose_generator
from utils.visualization import render_only, render_animation


def demo_visualize(mode, cfg, model, diffusion, dataset, actions):

    if cfg.dataset != 'h36m' and mode != 'pred':
        raise NotImplementedError(f"sorry, {mode} is currently only available in h36m setting.")

    if mode == 'joint':
#-0.070449 0.028574 0.988494
        """
        ['Phoning', 'Smoking 1', 'Photo 1', 'Greeting', 'Directions', 'Walking', 'Greeting 1', 
        'Purchases 1', 'WalkDog', 'Waiting', 'WalkTogether 1', 'Phoning 1', 'Posing 1', 'WalkTogether', 
        'WalkDog 1', 'Walking 1', 'Sitting', 'Purchases', 'SittingDown', 'Photo', 'SittingDown 1', 'Eating 1', 
        'Discussion 2', 'Discussion 1', 'Eating', 'Directions 1', 'Posing', 'Smoking', 'Sitting 1', 'Waiting 1']
        # actions = ['Walking', 'Walking']
        """
        """
        這裏data0[..., 1:, :]是把第一個點hip去掉了
        """
        data0 = dataset['test'].my_sample_iter_action(actions[0], cfg.dataset, sample_mode='last')
        data1 = dataset['test'].my_sample_iter_action(actions[1], cfg.dataset, sample_mode='first')
        data0 = data0[..., 1:, :].reshape([data0.shape[0], cfg.t_his + cfg.t_pred + cfg.t_fut, -1])
        data1 = data1[..., 1:, :].reshape([data1.shape[0], cfg.t_his + cfg.t_pred + cfg.t_fut, -1])

        pose_gen = pose_generator(data_set=dataset['test'],
                                  model_select=model,
                                  diffusion=diffusion,
                                  cfg=cfg,
                                  mode='joint',
                                  action=actions,
                                  nrow=cfg.vis_row)


        #将his+pre+fut提取出来
        render_only(skeleton=dataset['test'].skeleton, poses_generator=pose_gen, algos=['HumanMAC'],
                         t_hist=cfg.t_his,
                         t_predt=cfg.t_pred,
                         action_0=actions[0],
                         action_1=actions[1],
                         traj_his=data0,
                         traj_fut=data1,
                         ncol=cfg.vis_col,
                         complete=True,
                         output=os.path.join('/home/ghy/workspace/MOCAP/myHumanMAC/extractGif/gif/', f'{actions[0]}_joint_{actions[1]}.gif'), mode=mode)

    else:
        raise


