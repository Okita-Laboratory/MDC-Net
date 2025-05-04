from utils import *
import numpy as np
from data_loader.dataset_h36m import DatasetH36M
from data_loader.dataset_h36m_multimodal import DatasetH36M_multi


if __name__=='__main__':
    t_his = 2
    t_pred = 5
    t_fut = 2
    print(list(range(t_his)))
    traj_np = np.ones((3, t_his+t_pred+t_fut, 3, 3))
    padding = 'Zero'
    idx_pad, zero_index = generate_pad(padding, t_his=t_his, t_pred=t_pred, t_fut=t_fut)
    print(f'idx_pad={idx_pad}, zero_index={zero_index}')

    traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], t_his + t_pred + t_fut, -1])
    print(f'traj_np.shape={traj_np.shape}')
    traj = traj_np
    traj_pad = padding_traj(traj, padding, idx_pad, zero_index)
    print(f'traj_pad.shape={traj_pad.shape}')
    mask = traj == traj_pad
    # print(mask)
    diff_indices = np.where(mask == False)
    for idx in zip(*diff_indices):
        print(f"不同的元素位置：{idx}，值分别为：{traj[idx]} 和 {traj_pad[idx]}")

    # test sampling
    # np.random.seed(0)
    # actions = {'WalkDog'}
    # dataset = DatasetH36M('train', actions=actions)
    # generator = dataset.sampling_generator()
    # dataset.normalize_data()
    # # generator = dataset.iter_generator()
    # for data in generator:
    #     print(data.shape)