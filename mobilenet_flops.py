from src.compute_flops import print_model_param_flops
from src.mobilenet import pruned_mobilenet
import torch


settings = []
settings.append([[1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],])
settings.append([[1, 16, 1, 1],  # 2 bn
            [6, 24, 2, 2],  # 6 bn
            [5, 32, 3, 2],  # 9 bn
            [4, 64, 3, 2],  # 12 bn
            [5, 96, 2, 1],  # 9 bn
            [3, 160, 2, 2],  # 9 bn
            [3, 320, 1, 1]])  # 3 bn]
settings.append([[1, 16, 1, 1],  # 2 bn
            [6, 24, 2, 2],  # 6 bn
            [5, 32, 3, 2],  # 9 bn
            [2, 64, 4, 2],  # 12 bn
            [2, 96, 3, 1],  # 9 bn
            [2, 160, 3, 2],  # 9 bn
            [3, 320, 1, 1]])  # 3 bn
settings.append([[1, 16, 1, 1],
         [6, 24, 3, 2],
         [6, 32, 5, 2],
         [6, 64, 5, 2],
         [6, 96, 5, 1],
         [6, 160, 3, 2],
         [6, 320, 1, 1]],)
settings.append([[1, 16, 1, 1],
                [4, 24, 3, 2],
                [3, 32, 4, 2],
                [3, 64, 2, 2],
                [2, 96, 2, 1],
                [2, 160, 2, 2],
                [4, 320, 1, 1]],)
settings.append([[1, 16, 1, 1],
         [8, 24, 2, 2],
         [8, 32, 3, 2],
         [10, 64, 4, 2],
         [10, 96, 3, 1],
         [8, 160, 3, 2],
         [6, 320, 1, 1]],)
settings.append([[1, 16, 1, 1],
                [5, 24, 2, 2],
                [5, 32, 3, 2],
                [4, 64, 2, 2],
                [3, 96, 2, 1],
                [2, 160, 2, 2],
                [4, 320, 1, 1]],  # modified by struct2
        )


class_num = 100


weights = []
# weight0 = ".idea/mobilenet_cifar100_.pth"
# weight1 = "finetune_result/mobilenet_0422/model/mobilenet_cifar100_1.pth"
# weight2 = "finetune_result/mobilenet_0422/model/mobilenet_cifar100_2.pth"

weights.append(".idea/mobilenet_cifar100_.pth")
weights.append("finetune_result/mobilenet_0422/model/mobilenet_cifar100_1.pth")
weights.append("finetune_result/mobilenet_0422/model/mobilenet_cifar100_2.pth")
weights.append("sparse_result/0426/model/mobilenet_cifar100__0.0002_struct1.pth")
weights.append("finetune_result/0426/model/mobilenet_cifar100_5.pth")
weights.append("sparse_result/0426/model/mobilenet_cifar100__0.0002_struct2.pth")
weights.append("finetune_result/0426/model/mobilenet_cifar100_6.pth")


def getflops(weight, setting):
    model = pruned_mobilenet.MobileNetV2(num_classes=class_num, inverted_residual_setting=setting)
    model.load_state_dict(torch.load(weight, map_location="cpu")["state_dict"])
    print("The flops of the model is ------>{}".format(print_model_param_flops(model)))


if __name__ == '__main__':
    for idx, (s, w) in enumerate(zip(settings, weights)):
        if idx == 0:
            print("Testing of origin models...")
        if idx == 3:
            print("\nTesting structure1")
        if idx == 5:
            print("\nTesting structure1")
        getflops(w, s)


# weight0 = ".idea/mobilenet_cifar100_.pth"
# weight1 = "finetune_result/mobilenet_0422/model/mobilenet_cifar100_1.pth"
# weight2 = "finetune_result/mobilenet_0422/model/mobilenet_cifar100_2.pth"
#
# model0 = pruned_mobilenet.MobileNetV2(num_classes=class_num, inverted_residual_setting=setting0)
# model1 = pruned_mobilenet.MobileNetV2(num_classes=class_num, inverted_residual_setting=setting1)
# model2 = pruned_mobilenet.MobileNetV2(num_classes=class_num, inverted_residual_setting=setting2)
#
# model0.load_state_dict(torch.load(weight0, map_location="cpu")["state_dict"])
# model1.load_state_dict(torch.load(weight1, map_location="cpu")["state_dict"])
# model2.load_state_dict(torch.load(weight2, map_location="cpu")["state_dict"])
#
# print("The flops of model0------>{}".format(print_model_param_flops(model0)))
# print("The flops of model1------>{}".format(print_model_param_flops(model1)))
# print("The flops of model2------>{}".format(print_model_param_flops(model2)))
