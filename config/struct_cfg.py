
cifar100_channel_prune = {
        "small":
        [[1, 8, 1, 1],
         [6, 12, 2, 2],
         [6, 12, 3, 2],
         [6, 17, 4, 2],
         [6, 22, 3, 1],
         [6, 33, 3, 2],
         [6, 315, 1, 1]],
        "middle":
        [[1, 8, 1, 1],
         [6, 13, 2, 2],
         [6, 19, 3, 2],
         [6, 26, 4, 2],
         [6, 32, 3, 1],
         [6, 42, 3, 2],
         [6, 315, 1, 1]],
        "big":
        [[1, 10, 1, 1],
         [6, 16, 2, 2],
         [6, 24, 3, 2],
         [6, 32, 4, 2],
         [6, 40, 3, 1],
         [6, 48, 3, 2],
         [6, 316, 1, 1]],
        "huge":
        [[1, 12, 1, 1],
         [6, 18, 2, 2],
         [6, 22, 3, 2],
         [6, 42, 4, 2],
         [6, 60, 3, 1],
         [6, 100, 3, 2],
         [6, 318, 1, 1]],
        "enormous":
        [[1, 14, 1, 1],
        [6, 20, 2, 2],
        [6, 28, 3, 2],
        [6, 48, 4, 2],
        [6, 72, 3, 1],
        [6, 120, 3, 2],
        [6, 318, 1, 1],]
}


times_prune = {
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
}

bigger_origin = {
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
    "origin_c_big_t":[
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
    ]
}


