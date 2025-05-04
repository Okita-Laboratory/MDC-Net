import argparse
import sys
import os
from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.freedom_pose_gen import freedom_pose_gen, sample_one_action
from utils.script import *
import ast
sys.path.append(os.getcwd())
from config import Config, update_config
import torch
from tensorboardX import SummaryWriter
from utils.training import Trainer
from utils.evaluation import compute_stats
from utils.visualization import render_skeleton

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        default='h36m', help='h36m or humaneva')
    parser.add_argument('--mode', default='train', help='joint/ train/ freedom/ render')

    parser.add_argument('--actions', default='', help=['Phoning', 'Smoking 1', 'Photo 1', 'Greeting', 'Directions', 'Walking', 'Greeting 1',
        'Purchases 1', 'WalkDog', 'Waiting', 'WalkTogether 1', 'Phoning 1', 'Posing 1', 'WalkTogether',
        'WalkDog 1', 'Walking 1', 'Sitting', 'Purchases', 'SittingDown', 'Photo', 'SittingDown 1', 'Eating 1',
        'Discussion 2', 'Discussion 1', 'Eating', 'Directions 1', 'Posing', 'Smoking', 'Sitting 1', 'Waiting 1'])
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    parser.add_argument('--milestone', type=list, default=[75, 150, 225, 275, 350, 450])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_model_interval', type=int, default=40)
    parser.add_argument('--save_gif_interval', type=int, default=20)
    parser.add_argument('--save_metrics_interval', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='/home/ghy/workspace/MOCAP/myHumanMAC//checkpoints/mymodel_cos_layer4.pt', help='')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_switch_num', type=int, default=10)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=3)
    parser.add_argument('--input', type=str, default='/home/ghy/workspace/MOCAP/output/final_pose/h36m_keypoints.txt')
    parser.add_argument('--sample_mode', type=str, default='all')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=str, default=500)
    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))
    print(cfg)

    dataset, dataset_multi_test = dataset_split(cfg)
    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    """model"""
    model, diffusion = create_model_and_diffusion(cfg)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
        # prepare full evaluation dataset
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        trainer = Trainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            logger=logger,
            tb_logger=tb_logger)
        trainer.loop()

    elif args.mode == 'joint':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        # 将 actions 字符串转换为列表（如果需要多个动作）
        actions = args.actions.split(',') if ',' in args.actions else [args.actions]
        # 去掉列表中每个元素的空格
        actions = [action.strip() for action in actions]
        demo_visualize(args.mode, cfg, model, diffusion, dataset, actions)
    elif args.mode == 'freedom':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        with open('/home/ghy/workspace/MOCAP/output/action_path.txt', 'r') as file:
            action_path = [line.strip() for line in file.readlines()]
        freedom_pose_gen(cfg, model, diffusion, dataset, action_path)
    elif args.mode == 'render':
        file_path = args.input
        data = np.loadtxt(file_path)
        num_frames = data.shape[0] // 17
        final_pose = data.reshape((num_frames, 17, 3))
        render_skeleton(final_pose, skeleton=dataset['test'].skeleton, output=file_path)
    elif args.mode == 'sample':
        action = args.actions
        sample_mode = args.sample_mode # all, part
        start = args.start
        end = args.end
        data = sample_one_action(cfg, dataset, action, sample_mode=sample_mode, start=start, end=end)