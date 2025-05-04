import numpy as np

if __name__ == '__main__':

    # model = MyMotionTransformer(input_feats=3 * 16)
    # print(model)

    array = np.ones((10, 125, 3))
    print(array.shape)

    indices = np.concatenate((np.arange(0, 25), np.arange(75, 125)))

    # 使用高级索引选择行
    gt_group = array[:, indices, :]
    print(gt_group.shape)

    action_list = dataset['test'].prepare_iter_action(cfg.dataset)