import os
import torch
import torch.nn as nn
import numpy as np
import config.mobilenet_cfg as config
from torchvision import models

device = "cpu"

models_path = config.prune_folder
model_path = [os.path.join(models_path, model_name) for model_name in os.listdir(models_path) if "sparse" in model_name]
cifar10_model, cifar100_model = [], []
for m_path in model_path:
    if "cifar100" in m_path:
        cifar100_model.append(m_path)
    elif "cifar10" in m_path:
        cifar10_model.append(m_path)

setting_dict = config.settings
percent_ls = config.prune_percent
only_threshold = config.prune_only_threshld
threshold_ls = config.prune_threshold


def obtain_prune_idx(path):
    lines = []
    with open(path, 'r') as f:
        file = f.readlines()
        for line in file:
            lines.append(line)
    idx = 0
    prune_idx = []
    for line in lines:
        if "):" in line:
            idx += 1
        if "BatchNorm2d" in line:
            # print(idx, line)
            prune_idx.append(idx)

    prune_idx = prune_idx[1:]  # 去除第一个bn1层
    return prune_idx


def sort_bn(model, prune_idx):
    size_list = [m.weight.data.shape[0] for idx, m in enumerate(model.modules()) if idx in prune_idx]
    # bn_layer = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_prune_layers = [m for idx, m in enumerate(model.modules()) if idx in prune_idx]
    bn_weights = torch.zeros(sum(size_list))

    index = 0
    for module, size in zip(bn_prune_layers, size_list):
        bn_weights[index:(index + size)] = module.weight.data.abs().clone()
        index += size
    sorted_bn = torch.sort(bn_weights)[0]
    return sorted_bn


def obtain_bn_mask(bn_module, thre):
    if device != "cpu":
        thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()
    return mask


def obtain_bn_threshold(model, sorted_bn, percentage):
    thre_index = int(len(sorted_bn) * percentage)
    thre = sorted_bn[thre_index]

    return thre


def obtain_filters_mask(model, prune_idx, thre, file_path):
    pruned = 0
    bn_count = 0
    total = 0
    num_filters = []
    pruned_filters = []
    filters_mask = []
    pruned_maskers = []

    for idx, module in enumerate(model.modules()):
        if isinstance(module, nn.BatchNorm2d):
            if idx in prune_idx:
                mask = obtain_bn_mask(module, thre).cpu().numpy()
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                if remain == 0:  # 保证至少有一个channel
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = module.weight.data.abs().max()
                    mask = obtain_bn_mask(module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain
                    bn_count += 1
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}', file=file_path)

                pruned_filters.append(remain)
                pruned_maskers.append(mask.copy())
            else:
                mask = np.ones(module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}', file=file_path)

    return pruned_filters, pruned_maskers


if __name__ == '__main__':
    for path in cifar10_model:
        struct = path.split("_")[-1][6:][:-4]
        model = models.mobilenet_v2(num_classes=10, inverted_residual_setting=setting_dict[struct])

        model_name = "./tmp.txt"
        print(model, file=open(model_name, 'w'))
        prune_idx = obtain_prune_idx(model_name)

        model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        sorted_bn = sort_bn(model, prune_idx)
        dest_folder = os.path.join("prune_results", path.split("/")[1], "cifar10", path.split("\\")[-1][:-4])
        os.makedirs(dest_folder, exist_ok=True)
        if only_threshold:
            file = open(os.path.join("prune_results", path.split("/")[1], path.split("\\")[-1][:-4] + ".txt"), "w")
            print("Percent,      Threshold", file=file)
            for thresh in threshold_ls:
                threshold = obtain_bn_threshold(model, sorted_bn, thresh)
                print("{},      {}".format(thresh, threshold), file=file)
        else:
            for percent in percent_ls:
                file = open(os.path.join(dest_folder, "{}.txt".format(percent)), "w")
                threshold = obtain_bn_threshold(model, sorted_bn, percent)
                file.write("threshold is {}\n\n".format(threshold))
                pruned_filters, pruned_maskers = obtain_filters_mask(model, prune_idx, threshold, file)

    for path in cifar100_model:
        # struct = path.split("_")[-1][6:][:-4]
        struct = "_".join(path.split("_")[-4:])[6:-4]
        model = models.mobilenet_v2(num_classes=100, inverted_residual_setting=setting_dict[struct])

        model_name = "./tmp.txt"
        print(model, file=open(model_name, 'w'))
        prune_idx = obtain_prune_idx(model_name)

        model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        sorted_bn = sort_bn(model, prune_idx)
        dest_folder = os.path.join("prune_results", path.split("/")[1], "cifar100", path.split("\\")[-1][:-4])
        os.makedirs(dest_folder, exist_ok=True)
        if only_threshold:
            file = open(os.path.join("prune_results", path.split("/")[1], path.split("\\")[-1][:-4] + ".txt"), "w")
            print("Percent,      Threshold", file=file)
            for thresh in threshold_ls:
                threshold = obtain_bn_threshold(model, sorted_bn, thresh)
                print("{},      {}".format(thresh, threshold), file=file)
        else:
            for percent in percent_ls:
                file = open(os.path.join(dest_folder, "{}.txt".format(percent)), "w")
                threshold = obtain_bn_threshold(model, sorted_bn, percent)
                file.write("threshold is {}\n\n".format(threshold))
                pruned_filters, pruned_maskers = obtain_filters_mask(model, prune_idx, threshold, file)


