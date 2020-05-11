import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import config.mobilenet_cfg as config
import time
from tensorboardX import SummaryWriter
import torchvision.models as models
from src.compute_flops import print_model_param_flops, print_model_param_nums
import numpy as np
# from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
# signal(SIGPIPE, SIG_IGN)

try:
    from apex import amp
    mixed_precision = True
except:
    mixed_precision = False

sparse_name = {True: "sparse", False: ""}

seed = 1
start_epoch = 0
dest_folder = os.path.join("sparse_result", config.mob_dest)
os.makedirs(os.path.join(dest_folder, "model"), exist_ok=True)
os.makedirs(os.path.join(dest_folder, "log"), exist_ok=True)
bn_log_dest = os.path.join("bn_log", config.mob_dest)
os.makedirs(bn_log_dest, exist_ok=True)

lr = config.lr
device = config.device
num_workers = config.num_workers
weight_decay = 1e-4
momentum = 0.9
sparse_param_ls = config.mob_sparse_param

epochs = config.mob_epoch
batch_size = config.mob_batch_size
datasetss = config.mob_sparse_dataset
ifsparse_dict = config.mob_ifsparse


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class MobilenetSparseTrainer:
    def __init__(self, model_type, sparse, dataset, sparse_s, setting, s_idx):
        self.batch_size = batch_size
        self.sparse = sparse
        self.epoch = config.mob_epoch
        self.sparse_s = sparse_s
        self.model_name = model_type
        self.count_batch = 0
        self.model_str = "{}_{}_{}_{}_struct{}.pth".format(model_type, dataset, sparse_name[sparse], sparse_s, s_idx)
        log_str = self.model_str.replace("pth", "txt")
        self.log = open(os.path.join(dest_folder, "log", log_str), "w")
        self.num_classes, self.train_loader, self.test_loader = self.get_dataset(dataset)
        self.model = models.mobilenet_v2(num_classes=self.num_classes, inverted_residual_setting=setting)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.model = self.model.to(device)

        self.bn_file = open(os.path.join(bn_log_dest, self.model_str.replace(".pth", "_bn.txt")), "w")

        if mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        os.makedirs(os.path.join("runs", dest_folder), exist_ok=True)
        self.writer = SummaryWriter(os.path.join("runs", dest_folder, self.model_str[:-4]))
        self.model_path = os.path.join(dest_folder, "model", self.model_str)
        os.makedirs(self.model_path[:-4], exist_ok=True)
        self.save_structure()

    def get_dataset(self, ds):
        if ds == "cifar10":
            trainset = datasets.CIFAR10(root='./data.cifar10', train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR10(root='./data.cifar10', train=False, download=True, transform=transform_test)
            nums = 10
        elif ds == "cifar100":
            trainset = datasets.CIFAR100(root='./data.cifar100', train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR100(root='./data.cifar100', train=False, download=True, transform=transform_test)
            nums = 100
        else:
            raise ValueError()
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=5)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=2)
        return nums, train_loader, test_loader

    def save_structure(self):
        os.makedirs("structure", exist_ok=True)
        structure_file = "structure/{}.txt".format(self.model_name)
        if not os.path.isfile(structure_file):
            print(self.model, file=open(structure_file, 'w'))

    def updateBN(self):
        bn_sum, bn_num = 0, 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(self.sparse_s * torch.sign(m.weight.data))  # L1
                bn_sum += sum(abs(m.weight.grad.data)).cpu().tolist()
                bn_num += len(m.weight.grad.data)
        return bn_sum/bn_num

    def calBN(self):
        bn_sum, bn_num = 0, 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_sum += sum(abs(m.weight.grad.data)).cpu().tolist()
                bn_num += len(m.weight.grad.data)
        return bn_sum/bn_num

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                # test_loss += nn.CrossEntropyLoss(output, target, reduction='sum').item()
                test_loss += F.cross_entropy(output, target, reduction='sum').item()

                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(self.test_loader.dataset)
            print('Test set: learning rate : {} Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(lr,
                     test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))
            self.log.write('\nTest set: learning rate : {} Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(lr,
                     test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))

        return float(correct) / float(len(self.test_loader.dataset)), test_loss

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.count_batch += 1
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            # train_loss += loss
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            if mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.sparse:
                bn = self.updateBN()
            else:
                bn = self.calBN()
            self.writer.add_scalar("bn", bn, self.count_batch)
            print("{}:{}".format(self.count_batch, bn), file=self.bn_file)
            self.optimizer.step()

            train_loss /= len(self.train_loader.dataset)
            if batch_idx % 100 == 0 or batch_idx + 1 == len(self.train_loader):
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\t lr:{} Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), lr, loss.item()))
                self.log.write('Train Epoch: {} [{}/{} ({:.1f}%)]\t lr:{} Loss: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), lr, loss.item()))
        return float(correct) / float(len(self.train_loader.dataset)), train_loss

    def save_checkpoint(self, state, is_best, epoch, filepath):
        torch.save(state, filepath)

    def train_model(self):
        begin = time.time()
        best_prec, best_epoch = 0., 0.
        # self.writer.add_graph(self.model, torch.rand(1, 3, 32, 32).to(device))
        best_train_loss, best_train_acc, best_val_loss, best_val_acc = float("inf"), 0, float("inf"), 0

        flops = print_model_param_flops(self.model)
        params = print_model_param_nums(self.model)
        self.log.write("The flops is {}\n".format(flops))
        self.log.write("The params is {}\n".format(params))
        print("The flops is {}\n".format(flops))
        print(("The params is {}\n".format(params)))

        for epoch in range(start_epoch, self.epoch):
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(
                    name, param.clone().data.to("cpu").numpy(), epoch)

            bn_ls = [m for m in list(self.model.modules()) if isinstance(m, nn.BatchNorm2d)]
            bn_all_weight = []
            for idx, bn_layer in enumerate(bn_ls):
                bn_weight = list(bn_layer.parameters())[0]
                bn_all_weight += bn_weight.tolist()
                self.writer.add_histogram("bn{}_weight".format(idx), bn_weight, epoch)
            self.writer.add_histogram("bn_all_weight", np.array(bn_all_weight), epoch)

            if epoch in [self.epoch * 0.5, self.epoch * 0.75]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
            elif epoch in [self.epoch * 0.75, self.epoch * 0.9]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.01
            elif epoch in [self.epoch * 0.9, self.epoch]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.001
            train_acc, train_loss = self.train(epoch)
            val_acc, val_loss = self.test()
            self.bn_file.write("\n")

            best_train_acc = train_acc if train_acc > best_train_acc else best_train_acc
            best_train_loss = train_loss if train_loss < best_train_loss else best_train_loss
            best_val_acc = val_acc if val_acc > best_val_acc else best_val_acc
            best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss

            self.writer.add_scalar("lr", lr, epoch)
            self.writer.add_scalar("train_acc", train_acc, epoch)
            self.writer.add_scalar("train_loss", train_loss, epoch)
            self.writer.add_scalar("val_acc", val_acc, epoch)
            self.writer.add_scalar("val_loss", val_loss, epoch)

            torch.save(self.model.state_dict(), os.path.join(self.model_path[:-4], "model_{}.pth".format(epoch)))
            is_best = val_acc > best_prec
            if val_acc > best_prec:
                best_epoch = epoch
                best_prec = val_acc
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_prec': best_prec,
                    'best_epoch': best_epoch,
                    'optimizer': self.optimizer.state_dict(),
                    'learning_rate': lr,
                }, is_best, epoch, filepath=self.model_path)

        print("Best epoch: {} accuracy: {}".format(best_epoch, best_prec))
        self.log.write("Best epoch: {} accuracy: {}\n".format(best_epoch, best_prec))
        self.log.write("Total time cost is {}".format(round(time.time()-begin), 2))

        result = "{},{},{},{},{},{},{},{}".format(self.model_str, params, flops, best_train_acc, best_train_loss,
                                                     best_val_acc, best_val_loss, best_epoch)
        return result


def record_info():
    with open(os.path.join(dest_folder, "record.txt"), "a+") as f:
        out = "network: mobilenet"
        f.write(out + '\n')
        out = "dataset: "
        for ds in datasetss:
            out = out + ds + "\t"
        f.write(out + '\n')
        f.write("sparse: True, False\n")
        f.write("optimizer: {}\n".format(config.optimizer))
        f.write("lr: {}\n".format(lr))
        f.write("decay: {}\n".format(weight_decay))


def record_res(result):
    with open(os.path.join(dest_folder, "record.txt"), "a+") as f:
        f.write(result)
        f.write("\n")


if __name__ == '__main__':
    record_info()
    for dset in datasetss:
        for setting_idx in config.sparse_setting:
            for if_sparse in ifsparse_dict:
                trained_unsparse = False
                for sp in sparse_param_ls:
                    if not if_sparse:
                        if not trained_unsparse:
                            trained_unsparse = True
                            print("\n\n\n-------------->Training {} with {}, sparse is {}, sp is {}, setting idx is {}".
                                  format("mobilenet", dset, if_sparse, sp, setting_idx))
                            MST = MobilenetSparseTrainer("mobilenet", if_sparse, dset, sp, config.settings[setting_idx],
                                                         setting_idx)
                            res = MST.train_model()
                            record_res(res)
                    else:
                        print("\n\n\n-------------->Training {} with {}, sparse is {}, sp is {}, setting idx is {}".
                              format("mobilenet", dset, if_sparse, sp, setting_idx))
                        MST = MobilenetSparseTrainer("mobilenet", if_sparse, dset, sp, config.settings[setting_idx],
                                                     setting_idx)
                        res = MST.train_model()
                        record_res(res)
