import torch
import numpy
import copy
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

from datetime import datetime

colors_16 = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
             [255, 255, 0], [170, 255, 0], [85, 255, 0],
             [0, 255, 0], [0, 255, 85], [0, 255, 170],
             [0, 255, 255], [0, 170, 255], [0, 85, 255],
             [0, 0, 255], [85, 0, 255], [170, 0, 255],
             [255, 0, 255]]


def render_only(skeleton, poses_generator, algos,
                     t_hist, t_predt, traj_his, traj_fut, action_0=None, action_1=None,
                     complete=False, fix_0=True, azim=0.0, output=None, mode='pred', size=2, ncol=5, bitrate=3000, fix_index=None):
    # action_0 is action_his's name
    # action_1 is action fut's name

    if mode == 'joint':
        fix_0 = False
    if fix_index is not None:
        fix_list = [
            [1, 2, 3],  #
            [4, 5, 6],
            [7, 8, 9, 10],
            [11, 12, 13],
            [14, 15, 16],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        fix_i = fix_list[fix_index]
        fix_col = 'darkblue'
    else:
        fix_i = None
    all_poses = next(poses_generator)
    del all_poses['HumanMAC_10']
    del all_poses['HumanMAC_11']
    t_125 = 125
    if complete == True:

        # 完整的his+pre+fut
        temp1 = all_poses['HumanMAC_12'][t_hist : (t_hist + t_predt), :, :]
        #temp1.shape:(90, 17, 3)
        traj_his = np.pad(torch.tensor(traj_his).squeeze().reshape((t_125, 16, 3)), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
        #print(traj_his.shape)(125, 17, 3)
        traj_fut = np.pad(torch.tensor(traj_fut).squeeze().reshape((t_125, 16, 3)), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
        #print(traj_fut.shape)(125, 17, 3)
        # temp is the complete motion sequence his+pre+fut
        # temp = np.concatenate((traj_his, np.zeros((125, 17, 3)), traj_fut), 0)
        temp = np.concatenate((traj_his, temp1, traj_fut), 0)
        tmep_copy = copy.deepcopy(temp)
        tmep_copy[:, :, [2]] = - tmep_copy[:, :, [2]]
        tmep_copy[:, :, [0, 1, 2]] = tmep_copy[:, :, [0, 2, 1]]
        # print('temp=', tmep_copy.shape)temp= (340, 17, 3)
        # 指定输出路径
        # action_0_cleaned = action_0.replace(" ", "_")
        # action_1_cleaned = action_1.replace(" ", "_")
        output_path = f"/home/ghy/workspace/MOCAP/output/{action_0}2{action_1}/h36m_keypoints.txt"  # 请将此路径替换为您希望保存的实际路径
        # 保存为 txt 文件
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(output_path, tmep_copy.reshape(-1, 3), fmt='%.6f')
        print(f"已保存到: {output_dir}")


        '''
            保存每个点的座标值(x,y,z)txt里面。
        '''
        bodyName = ['hip', 'r_hip', 'r_knee', 'r_foot', 'l_hip', 'l_knee', 'l_foot', 'spine', 'thorax',
                'neck', 'head', 'l_shoulder', 'l_elbow', 'l_wrist', 'r_shoulder', 'r_elbow', 'r_wrist']
        # 仅输出的部位名称
        selected_body_parts = ['l_wrist']  # 指定输出的部位，可以根据需要修改

        timeStamp = np.arange(340).reshape(-1, 1) * 0.02
        print('timeStamp', timeStamp.shape)

        # 找到所选部位在 bodyName 中的索引
        selected_indices = [bodyName.index(part) for part in selected_body_parts if part in bodyName]

        # 输出指定部位的数据
        for index in selected_indices:
            saveData = np.concatenate((timeStamp, temp[:, index, :]), axis=1)
            dir_txt = os.path.join('/home/ghy/workspace/MOCAP/output/', f'{action_0}2{action_1}',
                                   f'{bodyName[index]}_xyz.txt')
            os.makedirs(os.path.dirname(dir_txt), exist_ok=True)
            np.savetxt(dir_txt, saveData, fmt='%.2f ' + '%.6f ' * (saveData.shape[1] - 1))

        print(f'已保存选定部位的数据：{selected_body_parts}')


        all_poses['HumanMAC_12'] = temp

    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    # total time
    t_total = next(iter(all_poses.values())).shape[0]
    """important  save poses that need to be presented"""
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                        all_poses.items()))

    # 关闭matplotlib的交互模式
    plt.ioff()
    fig = plt.figure(figsize=(2, 3))
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.3
    for index, (title, data) in enumerate(poses.items()):

        ax = fig.add_subplot(1, 1, index + 1, projection='3d')

        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_axis_off()# 去除坐标轴
        ax.grid(False) # 去除背景网格
        ax.dist = 5.0

        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        # save 每帧的第一个点的xy坐标
        trajectories.append(data[:, 0, [0, 1]])

    #  only save value
    poses = list(poses.values())

    anim = None
    initialized = False
    animating = True
    find = 0

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            if fix_0 and n % ncol == 0 and i >= t_hist:
                continue
            # trajectories only use for 调整3D视图限制x,y,z
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]

            """ 
            调整3D视图限制x,y,z
            设置x轴的视图限制。这里使用了trajectories[n][i, 0]（第i帧的x坐标）为中心，
            通过加减radius / 2来定义视图的宽度，确保关注点在视图中央。
            """
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2 + trajectories[n][i, 2]])

        if not initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if n == 0:
                        if i < t_125 + t_hist:
                            col = '#4682B4'
                        elif i > t_125 + t_hist + t_predt:
                            col = '#FFA500'
                        else:
                            col = numpy.array(colors_16[j - 1]) / 255.0
                    else:
                        col = 'gray'
                    # draw skeletons
                    # ax.scatter(pos[j, 0], pos[j, 1], pos[j, 2], col)
                    """
                    该方法在3D空间中绘制两个关节之间的连线。每个关节的位置是通过poses[n][i]获得的，
                    其中n代表当前遍历到的子图索引，i是动画的当前帧索引
                    """
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=5.0))

            initialized = True
        else:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    if fix_0 and n % ncol == 0 and i >= t_hist:
                        continue

                    if n == 0:
                        if i < t_125 + t_hist:
                            col = '#4682B4'
                        elif i > t_125 + t_hist + t_predt:
                            col = '#FFA500'
                        else:
                            col = numpy.array(colors_16[j - 1]) / 255.0
                    else:
                        col = 'gray'

                    pos = poses[n][i]

                    # ax.scatter(pos[j, 0], pos[j, 1], pos[j, 2], col)
                    x_array = np.array([pos[j, 0], pos[j_parent, 0]])
                    y_array = np.array([pos[j, 1], pos[j_parent, 1]])
                    z_array = np.array([pos[j, 2], pos[j_parent, 2]])

                    lines_3d[n][j - 1][0].set_data_3d(x_array, y_array, z_array)
                    lines_3d[n][j - 1][0].set_color(col)

    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                            all_poses.items()))
        if x[0] in {'gt', 'context'}:
            for ax, title in zip(ax_3d, poses.keys()):
                ax.set_title(title, y=1.0, fontsize=12)
        if mode == 'switch':
            if x[0] in {algo + '_0'}:
                for ax, title in zip(ax_3d, poses.keys()):
                    ax.set_title('target', y=1.0, fontsize=12)

        poses = list(poses.values())

    def save_figs():
        nonlocal algo, find
        old_algo = algo
        os.makedirs('out_svg', exist_ok=True)
        suffix = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]
        os.makedirs('out_svg_' + suffix, exist_ok=True)
        for algo in algos:
            reload_poses()
            for i in range(0, t_total + 1, 10):
                if i == 0:
                    update_video(0)
                else:
                    update_video(i - 1)
                fig.savefig('out_svg_' + suffix + '/%d_%s_%d.svg' % (find, algo, i), transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 50
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='pillow')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)

    save()
    show_animation()
    # plt.show()  # fanren!!!!!
    plt.close()


def render_animation(skeleton, poses_generator, algos,
                     t_hist, t_predt, traj_his=None, traj_fut=None, action_0=None, action_1=None,
                     fix_0=True, azim=0.0, output=None, mode='pred', size=2, ncol=5,bitrate=3000, fix_index=None):

    dir_txt = output.replace("gif", "txt")

    if mode == 'switch' or mode == 'joint':
        fix_0 = False
    if fix_index is not None:
        fix_list = [
            [1, 2, 3],  #
            [4, 5, 6],
            [7, 8, 9, 10],
            [11, 12, 13],
            [14, 15, 16],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        fix_i = fix_list[fix_index]
        fix_col = 'darkblue'
    else:
        fix_i = None
    all_poses = next(poses_generator)
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    # total time
    t_total = next(iter(all_poses.values())).shape[0]
    """important  save poses that need to be presented"""
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                        all_poses.items()))
    # 关闭matplotlib的交互模式
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    if mode != 'joint':
        fig = plt.figure(figsize=(size * ncol, size * nrow))
    else:
        fig = plt.figure()
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        if mode != 'joint':
            ax = fig.add_subplot(nrow, ncol, index+1, projection='3d')
        else:
            ax = fig.add_subplot(1, 3, index+1, projection='3d')
        if index == 0:
            ax.set_title(action_0, y=1.0, fontsize=12)
        elif index == 1:
            ax.set_title(action_1, y=1.0, fontsize=12)
        else:
            ax.set_title("Joint result", y=1.0, fontsize=12)
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_xticklabels([]) # 将x轴的刻度标签设置为空，即不显示x轴上的刻度标签。
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        """change the title of fig"""
        # if mode == 'switch' or mode == 'joint':
        #     if index == 0:
        #         ax.set_title('target', y=1.0, fontsize=12)
        if mode == 'pred' or 'fix' in mode or mode == 'control' or mode == 'zero_shot':
            # if index == 0 or index == 1:
            #     ax.set_title(title, y=1.0, fontsize=12)
            # show all title
            ax.set_title(title, y=1.0, fontsize=12)
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        # save 每帧的第一个点的xy坐标
        trajectories.append(data[:, 0, [0, 1]])
    # 减少子图之间和周围的空白区域。
    fig.tight_layout(h_pad=15, w_pad=15)
    # 直接调整子图之间的间距以及子图距离图形边缘的距离
    fig.subplots_adjust(wspace=0.6, hspace=0.4)

    #  only save value
    poses = list(poses.values())


    anim = None
    initialized = False
    animating = True
    find = 0

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            if fix_0 and n % ncol == 0 and i >= t_hist:
                continue
            # trajectories only use for 调整3D视图限制x,y,z
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            # trajectories[n].shape: (125, 3) n = 0,1,2,3,4,5,6

            """ 
            调整3D视图限制x,y,z
            设置x轴的视图限制。这里使用了trajectories[n][i, 0]（第i帧的x坐标）为中心，
            通过加减radius / 2来定义视图的宽度，确保关注点在视图中央。
            """
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2  + trajectories[n][i, 2]])

        if not initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if i < t_hist or i > t_hist + t_predt:
                    col = 'gray'
                else:
                    col = numpy.array(colors_16[j - 1]) / 255.0


                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]

                    if mode == 'joint':
                        if n == 2:
                            if i < t_hist or i > t_hist + t_predt:
                                col = 'gray'
                            else:
                                col = numpy.array(colors_16[j - 1]) / 255.0
                        else:
                            col = 'gray'
                    # draw skeletons
                    #ax.scatter(pos[j, 0], pos[j, 1], pos[j, 2], col)
                    """
                    该方法在3D空间中绘制两个关节之间的连线。每个关节的位置是通过poses[n][i]获得的，
                    其中n代表当前遍历到的子图索引，i是动画的当前帧索引
                    """
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=3.0))

            initialized = True
        else:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue


                if i < t_hist or i > t_hist + t_predt:
                    col = 'gray'
                else:
                    col = numpy.array(colors_16[j - 1]) / 255.0


                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    if fix_0 and n % ncol == 0 and i >= t_hist:
                        continue

                    if mode == 'joint':
                        if n == 2:
                            if i < t_hist or i > t_hist + t_predt:
                                col = 'gray'
                            else:
                                col = numpy.array(colors_16[j - 1]) / 255.0
                        else:
                            col = 'gray'

                    pos = poses[n][i]

                    #ax.scatter(pos[j, 0], pos[j, 1], pos[j, 2], col)
                    x_array = np.array([pos[j, 0], pos[j_parent, 0]])
                    y_array = np.array([pos[j, 1], pos[j_parent, 1]])
                    z_array = np.array([pos[j, 2], pos[j_parent, 2]])

                    lines_3d[n][j - 1][0].set_data_3d(x_array, y_array, z_array)
                    lines_3d[n][j - 1][0].set_color(col)

    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                            all_poses.items()))
        if x[0] in {'gt', 'context'}:
            for ax, title in zip(ax_3d, poses.keys()):
                ax.set_title(title, y=1.0, fontsize=12)
        if mode == 'switch':
            if x[0] in {algo + '_0'}:
                for ax, title in zip(ax_3d, poses.keys()):
                    ax.set_title('target', y=1.0, fontsize=12)
        
        poses = list(poses.values())
    
    def save_figs():
        nonlocal algo, find
        old_algo = algo
        os.makedirs('out_svg', exist_ok=True)
        suffix = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]
        os.makedirs('out_svg_' + suffix, exist_ok=True)
        for algo in algos:
            reload_poses()
            for i in range(0, t_total + 1, 10):
                if i == 0:
                    update_video(0)
                else:
                    update_video(i - 1)
                fig.savefig('out_svg_' + suffix + '/%d_%s_%d.svg' % (find, algo, i), transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 50
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='pillow')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    save()
    show_animation()
    #plt.show()  # fanren!!!!!
    plt.close()

def render_skeleton(pose, skeleton, output, azim=0, bitrate=3000, fps=50):
    bodyName = [
        'hip', 'r_hip', 'r_knee', 'r_foot', 'l_hip', 'l_knee', 'l_foot',
        'spine', 'thorax', 'neck', 'head', 'l_shoulder', 'l_elbow', 'l_wrist',
        'r_shoulder', 'r_elbow', 'r_wrist'
    ]
    skin_color = '#FFE0BD'  # 皮肤
    shirt_color = '#ADD8E6'  # 浅蓝色
    point_styles = {
        1: {'color': '#FF0000' , 'size': 2},  # hip
        2: {'color': '#FF5500', 'size': 2},  # r_hip
        3: {'color': '#FFAA00' , 'size': 2},  # r_knee
        4: {'color': '#FFFF00' , 'size': 2},  # r_foot
        5: {'color': '#AAFF00', 'size': 2},  # l_hip
        6: {'color': '#55FF00' , 'size': 2},  # l_knee
        7: {'color': '#00FF00', 'size': 2},  # l_foot
        8: {'color': '#00FF55', 'size': 2},  # spine
        9: {'color': '#00FFAA', 'size': 2},  # thorax
        10: {'color': '#00FFFF' , 'size': 2},  # neck
        11: {'color': '#00AAFF', 'size': 2},  # head
        12: {'color': '#0055FF', 'size': 2},  # l_shoulder
        13: {'color': '#0000FF', 'size': 2},  # l_elbow
        14: {'color': '#5500FF' , 'size': 2},  # l_wrist
        15: {'color': '#AA00FF', 'size': 2},  # r_shoulder
        16: {'color': '#FF00FF', 'size': 2},  # r_elbow
        17: {'color': '#00FF00', 'size': 2},  # r_wrist
    }
    line_styles = {
        1: {'color': '#FF0000', 'linewidth': 5.0},  # hip -> r_hip
        2: {'color': '#FF5500', 'linewidth': 5.0},  # r_hip -> r_knee
        3: {'color': '#FFAA00', 'linewidth': 5.0},  # r_knee -> r_foot
        4: {'color': '#FFFF00', 'linewidth': 5.0},  # hip -> l_hip
        5: {'color': '#AAFF00', 'linewidth': 5.0},  # l_hip -> l_knee
        6: {'color': '#55FF00', 'linewidth': 5.0},  # l_knee -> l_foot
        7: {'color': '#00FF00', 'linewidth': 5.0},  # hip -> spine
        8: {'color': '#00FF55', 'linewidth': 5.0},  # spine -> thorax
        9: {'color': '#00FFAA', 'linewidth': 5.0},  # thorax -> neck
        10: {'color': '#00FFFF', 'linewidth': 5.0},  # neck -> head
        11: {'color': '#00AAFF', 'linewidth': 5.0},  # thorax -> l_shoulder
        12: {'color': '#0055FF', 'linewidth': 5.0},  # l_shoulder -> l_elbow
        13: {'color': '#0000FF', 'linewidth': 5.0},  # l_elbow -> l_wrist
        14: {'color': '#5500FF', 'linewidth': 5.0},  # thorax -> r_shoulder
        15: {'color': '#AA00FF', 'linewidth': 5.0},  # r_shoulder -> r_elbow
        16: {'color': '#FF00FF', 'linewidth': 5.0},  # r_elbow -> r_wrist
    }
    output_path = os.path.dirname(output) + '/render.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.ioff()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=azim)
    radius = 1.3
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([-radius / 2, radius / 2])
    ax.set_axis_off()

    parents = skeleton.parents()
    t_total = pose.shape[0]

    # 初始化线条和点列表
    lines_3d = []
    points_3d = []

    # 默认样式
    default_line_style = {'color': 'black', 'linewidth': 2.0}
    default_point_style = {'color': 'black', 'size': 20}

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        # 获取自定义线样式
        line_style = line_styles.get(j, default_line_style) if line_styles else default_line_style
        line, = ax.plot([], [], [], zdir='z', c=line_style['color'], linewidth=line_style['linewidth'])
        lines_3d.append(line)

        # 获取自定义点样式
        point_style = point_styles.get(j, default_point_style) if point_styles else default_point_style
        point = ax.scatter([], [], [], s=point_style['size'], c=point_style['color'])
        points_3d.append(point)

    def update_video(i):
        """更新每一帧骨骼位置"""
        pos = pose[i]
        for j, j_parent in enumerate(parents):
            if j_parent == -1:
                continue

            # 更新线的位置
            x_array = np.array([pos[j, 0], pos[j_parent, 0]])
            y_array = np.array([pos[j, 1], pos[j_parent, 1]])
            z_array = np.array([pos[j, 2], pos[j_parent, 2]])
            lines_3d[j - 1].set_data_3d(x_array, y_array, z_array)
        frame_dir = os.path.dirname(output) + '/imgs'
        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        plt.savefig(frame_path, dpi=100)

        # 更新点的位置
        for j, point in enumerate(points_3d):
            point._offsets3d = (
                [pos[j, 0]],
                [pos[j, 1]],
                [pos[j, 2]],
            )

    # 创建动画
    with tqdm(total=t_total, desc="Rendering frames") as pbar:
        def update_and_progress(i):
            update_video(i)
            pbar.update(1)

        anim = FuncAnimation(fig, update_and_progress, frames=np.arange(0, t_total), interval=1000 / fps, repeat=False)

    # 保存动画为MP4
    Writer = writers['ffmpeg']
    writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
    anim.save(output_path, writer=writer)

    print(f"视频已保存到 {output_path}!")
    plt.close()