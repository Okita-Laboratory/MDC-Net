import numpy as np


def iter_generator(step=25, n_modality=10):
    candi_tmp = np.ones((18627, 150, 17, 3))
    # candi_tmp.shape=(18627, 125, 17, 3)
    seq = np.random.rand(2356, 17, 3)
    # seq.shape=(2356, 17, 3)
    seq_len = seq.shape[0]
    for i in range(0, seq_len - t_total, step):
        traj = seq[None, i: i + t_total]
        # traj.shape=(1, 150, 17, 3)
        if n_modality > 0:
            margin_f = 1
            thre_his = 0.05
            thre_pred = 0.1
            x0 = np.copy(traj)
            # x0.shape=(1, 150, 17, 3)
            x0[:, :, 0] = 0
            # observation distance
            """
            x0.shape=(1, 150, 17, 3)
            candi_tmp.shape=(18627, 125, 17, 3)
            x0[:, self.t_his - margin_f:self.t_his, 1:].shape=(1, 1, 16, 3)
            candi_tmp[:, self.t_his - margin_f:self.t_his, 1:].shape=(18627, 1, 16, 3)
            """
            print(x0[:, t_his - margin_f:t_his, 1:].shape)
            print(candi_tmp[:, t_his - margin_f:t_his, 1:].shape)
            dist_his = np.mean(np.linalg.norm(x0[:, t_his - margin_f:t_his, 1:] -
                                              candi_tmp[:, t_his - margin_f:t_his, 1:], axis=3),
                               axis=(1, 2))
            idx_his = np.where(dist_his <= thre_his)[0]

            # future distance 平均欧氏距离
            dist_pred = np.mean(np.linalg.norm(x0[:, t_his:, 1:] -
                                               candi_tmp[idx_his, t_his:, 1:], axis=3), axis=(1, 2))

            idx_pred = np.where(dist_pred >= thre_pred)[0]
            # idxs = np.intersect1d(idx_his, idx_pred)
            traj_multi = candi_tmp[idx_his[idx_pred]]
            if len(traj_multi) > 0:
                traj_multi[:, :t_his] = traj[:, :t_his]
                if traj_multi.shape[0] > n_modality:
                    # idxtmp = np.random.choice(np.arange(traj_multi.shape[0]), n_modality, replace=False)
                    # traj_multi = traj_multi[idxtmp]
                    traj_multi = traj_multi[:n_modality]
            traj_multi = np.concatenate(
                [traj_multi, np.zeros_like(traj[[0] * (n_modality - traj_multi.shape[0])])],
                axis=0)
        else:
            traj_multi = None

        #yield traj, traj_multi

if __name__=='__main__':
    # t_his = 15
    # t_pre = 100
    # t_fut = 10
    # t_total = t_his + t_pre + t_fut
    # iter_generator(step=t_his)
    for i in range(115,125):
        print(i)