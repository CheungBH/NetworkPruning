import torch

lr = 0.1
optimizer = "momentum"
num_workers = 2
device = 'cuda'
# print("Using {}".format(device))


mob_dest = "test"
mob_sparse_dataset = ["cifar100"]
mob_sparse_param = [1e-4, 1e-3, 5e-4, 2e-4]
mob_ifsparse = [True]
sparse_setting = ["origin"]
# sparse_setting = ["big_c_huge_t_chprune", "big_c_big_t_chprune", "big_c_origin_t_chprune", "origin_c_big_t_chprune",
#                   "origin_c_huge_t_chprune"]

mob_epoch = 160
mob_batch_size = 1024


settings = {
    "big_c_huge_t_chprune": [
        # t, c, n, s
        [1, 16, 1, 1],
        [10, 16, 2, 2],
        [10, 18, 3, 2],
        [10, 24, 4, 2],
        [10, 30, 3, 1],
        [10, 40, 3, 2],
        [10, 320, 1, 1],
    ],
    "big_c_big_t_chprune": [
        # t, c, n, s
        [1, 16, 1, 1],
        [8, 20, 2, 2],
        [8, 24, 3, 2],
        [8, 30, 4, 2],
        [8, 36, 3, 1],
        [8, 48, 3, 2],
        [8, 320, 1, 1],
    ],
    "big_c_origin_t_chprune": [
        # t, c, n, s
        [1, 18, 1, 1],
        [6, 24, 2, 2],
        [6, 28, 3, 2],
        [6, 40, 4, 2],
        [6, 48, 3, 1],
        [6, 60, 3, 2],
        [6, 320, 1, 1],
    ],
    "origin_c_big_t_chprune": [
        # t, c, n, s
        [1, 12, 1, 1],
        [8, 16, 2, 2],
        [8, 18, 3, 2],
        [8, 30, 4, 2],
        [8, 30, 3, 1],
        [8, 48, 3, 2],
        [8, 320, 1, 1],
    ],
    "origin_c_huge_t_chprune": [
        # t, c, n, s
        [1, 10, 1, 1],
        [10, 14, 2, 2],
        [10, 16, 3, 2],
        [10, 24, 4, 2],
        [10, 30, 3, 1],
        [10, 40, 3, 2],
        [10, 320, 1, 1],
    ],
    "big_c_huge_t": [
        # t, c, n, s
        [1, 24, 1, 1],
        [10, 32, 2, 2],
        [10, 48, 3, 2],
        [10, 80, 4, 2],
        [10, 120, 3, 1],
        [10, 200, 3, 2],
        [10, 320, 1, 1],
    ],
    "big_c_big_t": [
        # t, c, n, s
        [1, 24, 1, 1],
        [8, 32, 2, 2],
        [8, 48, 3, 2],
        [8, 80, 4, 2],
        [8, 120, 3, 1],
        [8, 200, 3, 2],
        [8, 320, 1, 1],
    ],
    "big_c_origin_t": [
        # t, c, n, s
        [1, 24, 1, 1],
        [6, 32, 2, 2],
        [6, 48, 3, 2],
        [6, 80, 4, 2],
        [6, 120, 3, 1],
        [8, 200, 3, 2],
        [6, 320, 1, 1],
    ],
    "origin_c_big_t": [
        # t, c, n, s
        [1, 16, 1, 1],
        [8, 24, 2, 2],
        [8, 32, 3, 2],
        [8, 64, 4, 2],
        [8, 96, 3, 1],
        [8, 160, 3, 2],
        [8, 320, 1, 1],
    ],
    "origin_c_huge_t": [
        # t, c, n, s
        [1, 16, 1, 1],
        [10, 24, 2, 2],
        [10, 32, 3, 2],
        [10, 64, 4, 2],
        [10, 96, 3, 1],
        [10, 160, 3, 2],
        [10, 320, 1, 1],
    ],
    "origin":
        [[1, 16, 1, 1],
         [6, 24, 2, 2],
         [6, 32, 3, 2],
         [6, 64, 4, 2],
         [6, 96, 3, 1],
         [6, 160, 3, 2],
         [6, 320, 1, 1]],
    1:
        [[1, 16, 1, 1],
         [6, 24, 3, 2],
         [6, 32, 5, 2],
         [6, 64, 5, 2],
         [6, 96, 5, 1],
         [6, 160, 3, 2],
         [6, 320, 1, 1]],
    2:
        [[1, 16, 1, 1],
         [8, 24, 2, 2],
         [8, 32, 3, 2],
         [10, 64, 4, 2],
         [10, 96, 3, 1],
         [8, 160, 3, 2],
         [6, 320, 1, 1]],
        "small_hm_c100":
        [[1, 8, 1, 1],
         [6, 12, 2, 2],
         [6, 12, 3, 2],
         [6, 17, 4, 2],
         [6, 22, 3, 1],
         [6, 33, 3, 2],
         [6, 315, 1, 1]],
        "middle_hm_c100":
        [[1, 8, 1, 1],
         [6, 13, 2, 2],
         [6, 19, 3, 2],
         [6, 26, 4, 2],
         [6, 32, 3, 1],
         [6, 42, 3, 2],
         [6, 315, 1, 1]],
        "big_hm_c100":
        [[1, 10, 1, 1],
         [6, 16, 2, 2],
         [6, 24, 3, 2],
         [6, 32, 4, 2],
         [6, 40, 3, 1],
         [6, 48, 3, 2],
         [6, 316, 1, 1]],
        "huge_hm_c100":
        [[1, 12, 1, 1],
         [6, 18, 2, 2],
         [6, 22, 3, 2],
         [6, 42, 4, 2],
         [6, 60, 3, 1],
         [6, 100, 3, 2],
         [6, 318, 1, 1]],
        "enormous_hm_c100":
        [[1, 14, 1, 1],
         [6, 20, 2, 2],
         [6, 28, 3, 2],
         [6, 48, 4, 2],
         [6, 72, 3, 1],
         [6, 120, 3, 2],
         [6, 318, 1, 1], ]
}



mob_cifar100_settings = {
    "cifar100_large":
        [[1, 16, 1, 1],
         [5, 24, 2, 2],
         [5, 24, 3, 2],
         [5, 40, 4, 2],
         [4, 64, 3, 1],
         [3, 104, 3, 2],
         [2, 320, 1, 1]],
    "cifar100_middle":
        [[1, 16, 1, 1],
         [5, 24, 2, 2],
         [5, 24, 3, 2],
         [5, 40, 3, 2],
         [4, 64, 2, 1],
         [3, 104, 3, 2],
         [2, 320, 1, 1]],
    "cifar100_small":
        [[1, 16, 1, 1],
         [5, 24, 2, 2],
         [5, 32, 3, 2],
         [5, 40, 3, 2],
         [3, 64, 2, 1],
         [2, 104, 3, 2],
         [2, 320, 1, 1]],
                0:
                [[1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]],
                1:
                [[1, 16, 1, 1],  # 2 bn
                [6, 24, 2, 2],  # 6 bn
                [5, 32, 3, 2],  # 9 bn
                [4, 64, 3, 2],  # 12 bn
                [5, 96, 2, 1],  # 9 bn
                [3, 160, 2, 2],  # 9 bn
                [3, 320, 1, 1]],  # 3 bn]
                2:
                [[1, 16, 1, 1],  # 2 bn
                [6, 24, 2, 2],  # 6 bn
                [5, 32, 3, 2],  # 9 bn
                [2, 64, 4, 2],  # 12 bn
                [2, 96, 3, 1],  # 9 bn
                [2, 160, 3, 2],  # 9 bn
                [3, 320, 1, 1]],  # 3 bn
                3:
                [[1, 16, 1, 1],
                [4, 24, 2, 2],
                [3, 32, 3, 2],
                [2, 32, 3, 2],
                [6, 64, 3, 1],
                [4, 80, 2, 2],
                [4, 320, 1, 1]],
                4:
                [[1, 16, 1, 1],
                [4, 24, 2, 2],
                [3, 32, 3, 2],
                [4, 64, 2, 2],
                [3, 96, 1, 1],
                [2, 160, 2, 2],
                [4, 320, 1, 1]],
                5:
                [[1, 16, 1, 1],
                [4, 24, 3, 2],
                [3, 32, 4, 2],
                [3, 64, 2, 2],
                [2, 96, 2, 1],
                [2, 160, 2, 2],
                [4, 320, 1, 1]],  # modified by struct1
                6:
                [[1, 16, 1, 1],
                [5, 24, 2, 2],
                [5, 32, 3, 2],
                [4, 64, 2, 2],
                [3, 96, 2, 1],
                [2, 160, 2, 2],
                [4, 320, 1, 1]],  # modified by struct2
                7:
                [[1, 16, 1, 1],
                 [5, 24, 2, 2],
                 [4, 32, 3, 2],
                 [4, 48, 4, 2],
                 [3, 72, 3, 1],
                 [3, 96, 3, 2],
                 [6, 320, 1, 1]],  # modified by struct0
                8:
                [[1, 16, 1, 1],  # 2 bn
                 [6, 24, 2, 2],  # 6 bn
                 [5, 32, 3, 2],  # 9 bn
                 [4, 64, 3, 2],  # 12 bn
                 [5, 96, 2, 1],  # 9 bn
                 [3, 160, 2, 2],  # 9 bn
                 [3, 320, 1, 1]],  # 3 bn]
                9:
                    [[1, 15, 1, 1],
                     [5, 24, 2, 2],
                     [4, 32, 3, 2],
                     [2, 64, 2, 2],
                     [2, 96, 1, 1],
                     [2, 160, 1, 2],
                     [1, 320, 1, 1]],    # new method, origin mobilenet
                10:  [[1, 15, 1, 1],
                     [6, 18, 2, 2],
                     [6, 24, 3, 2],
                     [6, 36, 2, 2],
                     [6, 40, 1, 1],
                     [6, 60, 1, 2],
                     [6, 320, 1, 1]],    # new method, origin mobilenet, no prune cut channels
            }

mob_cifar10_settings = {0:
                [[1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]],
                1:
                [[1, 16, 1, 1],  # 2 bn
                [5, 24, 2, 2],  # 6 bn
                [4, 32, 3, 2],  # 9 bn
                [3, 64, 4, 2],  # 12 bn
                [2, 96, 3, 1],  # 9 bn
                [2, 160, 3, 2],  # 9 bn
                [3, 320, 1, 1]],  # 3 bn]
                2:
                [[1, 16, 1, 1],
                 [3, 24, 3, 2],
                 [2, 32, 5, 2],
                 [1, 64, 1, 2],
                 [1, 96, 1, 1],
                 [1, 160, 1, 2],
                 [2, 320, 1, 1]],
                3:
                [[1, 16, 1, 1],
                 [4, 24, 2, 2],
                 [3, 32, 3, 2],
                 [2, 64, 1, 2],
                 [1, 96, 1, 1],
                 [1, 160, 1, 2],
                 [2, 320, 1, 1]],
            }


mob_finetune_dataset = "cifar100"
mob_settings_ls = ["cifar100_large", "cifar100_middle", "cifar100_small"]
if mob_finetune_dataset == "cifar10":
    mob_settings = mob_cifar10_settings
elif mob_finetune_dataset == "cifar100":
    mob_settings = mob_cifar100_settings
else:
    raise ValueError("Wrong dataset name")


# pruning config
prune_folder = "sparse_result/mob_bigger_model_c100/model"
prune_percent = [0.76, 0.79, 0.82]
prune_only_threshld = False
prune_threshold = [float(num/100) for num in range(40,100,1)]

if __name__ == '__main__':
    print(prune_threshold)
