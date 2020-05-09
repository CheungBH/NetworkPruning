import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import config.mobilenet_cfg as config
from src.mobilenet import pruned_mobilenet
import time
from tensorboardX import SummaryWriter

try:
    from apex import amp
    mixed_precision = True
except:
    mixed_precision = False

seed = 1
resume = ""
start_epoch = 0
dest_folder = os.path.join("finetune_result", config.mob_dest)
os.makedirs(os.path.join(dest_folder, "model"), exist_ok=True)
os.makedirs(os.path.join(dest_folder, "log"), exist_ok=True)

lr = config.lr
device = config.device
num_workers = config.num_workers
weight_decay = 1e-4
momentum = 0.9

dataset = config.mob_finetune_dataset
setting_idx = config.mob_settings_ls

if dataset == "cifar10":
    settings = {k:v for k,v in config.mob_cifar10_settings.items() if k in setting_idx}
elif dataset == "cifar100":
    settings = {k:v for k,v in config.mob_cifar100_settings.items() if k in setting_idx}
else:
    raise ValueError("Wrong datasets name")

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


class MobilenetFinetuner:
    def __init__(self, model_type, set_idx, setting):
        self.batch_size = config.mob_batch_size
        self.epoch = config.mob_epoch
        self.model_name = model_type
        self.num_classes, self.train_loader, self.test_loader = self.get_dataset(dataset)
        self.model = pruned_mobilenet.MobileNetV2(self.num_classes, inverted_residual_setting=setting)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.model = self.model.to(device)
        if mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        model_str = "{}_{}_{}.pth".format(model_type, dataset, set_idx)
        log_str = model_str.replace("pth", "txt")
        self.model_path = os.path.join(dest_folder, "model", model_str)
        self.log = open(os.path.join(dest_folder, "log", log_str), "w")
        os.makedirs(os.path.join("runs", dest_folder), exist_ok=True)
        self.writer = SummaryWriter(os.path.join("runs", dest_folder, model_str[:-4]))

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
                                                   num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=num_workers)
        return nums, train_loader, test_loader

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
                     test_loss, correct,len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))
            self.log.write('\nTest set: learning rate : {} Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(lr,
                     test_loss, correct,len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))

        return float(correct) / float(len(self.test_loader.dataset)), test_loss

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            train_loss += loss
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

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

        for epoch in range(start_epoch, self.epoch):

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(
                    name, param.clone().data.to("cpu").numpy(), epoch)

            # lr decay !
            if epoch in [self.epoch * 0.5, self.epoch * 0.75]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
            train_acc, train_loss = self.train(epoch)
            val_acc, val_loss = self.test()

            self.writer.add_scalar("lr", lr, epoch)
            self.writer.add_scalar("train_acc", train_acc, epoch)
            self.writer.add_scalar("train_loss", train_loss, epoch)
            self.writer.add_scalar("val_acc", val_acc, epoch)
            self.writer.add_scalar("val_loss", val_loss, epoch)

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
        self.log.write("Best epoch: {} accuracy: {}".format(best_epoch, best_prec))
        self.log.write("Total time cost is {}".format(round(time.time()-begin), 2))


def record():
    with open(os.path.join(dest_folder, "record.txt"), "w") as f:
        out = "network: "
        f.write(out + '\n')
        out = "dataset: "
        f.write(out + '\n')
        f.write("sparse: True, False")
        f.write("optimizer: {}".format(config.optimizer))
        f.write("lr: {}".format(lr))
        f.write("decay: {}".format(weight_decay))


if __name__ == '__main__':
    for idx, sets in settings.items():
        print("Processing structure {}".format(idx))
        MF = MobilenetFinetuner('mobilenet', idx, sets)
        MF.train_model()
