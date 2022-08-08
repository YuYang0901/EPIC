import argparse
import logging
import os
import random
import shutil
import time

import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import models.vgg as vgg
import models.resnet as resnet
import models.ap_resnet as ap_resnet

from utils import AverageMeter, accuracy
from utils.subset import get_coreset, get_random_subset

from dataset.poison import PoisonedDataset
from dataset.cifar import get_target, cifar10_mean, cifar10_std
from dataset.tinyimagenet_module import TinyImageNet, tinyimagenet_mean, tinyimagenet_std


logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser(description='PyTorch EPIC Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'tinyimagenet'],
                        help='dataset name')
    parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'vgg16'],
                        help='dataset name')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of epochs', choices=(200, 40, 80))
    parser.add_argument('--batch-size', default=128, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--no_aug', action='store_true', help='do not use augmentation')
    parser.add_argument('--ap_model', action='store_true', help='use resnet in ap code')

    # Poison Setting
    parser.add_argument('--clean', action='store_true', help='train with the clean data')
    parser.add_argument("--poisons_path", type=str, help="where are the poisons?")
    parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
    parser.add_argument('--trigger_path', type=str, default=None, help='path to the trigger')
    parser.add_argument("--backdoor", action='store_true', help='whether we are using backdoor attack')
    
    # Medoid Selection
    parser.add_argument('--greedy', default='LazyGreedy', choices=('LazyGreedy', 'StochasticGreedy'), 
                        help='optimizer for subset selection')
    parser.add_argument('--metric', default='euclidean', choices=('euclidean', 'cosine'), 
                        help='metric for subset selection')
    parser.add_argument('--subset_size', '-s', dest='subset_size', type=float, help='size of the subset', default=1.0)
    parser.add_argument('--subset_sampler', type=str, default='coreset-pred', choices=('random', 'coreset-pred'), help='algorithm to select subsets')
    parser.add_argument('--subset_freq', type=int, default=10, help='frequency to update the subset')
    parser.add_argument('--equal_num', default=False, help='select equal numbers of examples from different classes')
    parser.add_argument('--scenario', default='scratch', choices=('scratch', 'transfer', 'finetune'), help='select the training setting')
    parser.add_argument('--top_frac', type=float, default=0.1, help='fraction of low-confidence poisons to keep/drop')

    # Data Pruning
    parser.add_argument('--cluster_thresh', type=float, default=1., help='thrshold to drop examples in the subset')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--drop_after', type=int, default=1, help='dropping small clusters after this epoch')
    parser.add_argument('--stop_after', type=int, default=200, help='stop dropping small clusters after this epoch')
    parser.add_argument('--drop_mile', action='store_true', help='dropping small clusters at specific epoch')


    args = parser.parse_args()

    global best_acc, transform_train, transform_val

    if args.dataset == 'tinyimagenet':
        crop_size = 64
        mean = tinyimagenet_mean
        std = tinyimagenet_std
    else:
        crop_size = 32
        mean = cifar10_mean
        std = cifar10_std

    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=crop_size,
                                  padding=4,
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def create_model(args):
        if args.arch == 'resnet18':
            if args.ap_model:
                model = ap_resnet.resnet_picker('ResNet18', 'CIFAR10')
            else:
                model = resnet.resnet18(num_classes=args.num_classes)
        elif args.arch == 'vgg16':
            model = vgg.vgg16(num_classes=args.num_classes)
        else:
            raise NotImplementedError

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    if args.seed is not None:
        set_seed(args)

    if args.scenario != 'scratch':
        args.resume = 'clean_models/resnet18-cifar10-200epochs.pth.tar'
        args.epochs = 40
        if args.scenario == 'transfer':
            args.stop_after = 10
        dataset_index = args.poisons_path.split('/')[-4]
        try:
            dataset_index = int(dataset_index)
        except:
            dataset_index = args.poisons_path.split('/')[-2]
    else:
        dataset_index = args.poisons_path.split('/')[-2]

    if args.epochs == 200:
        args.steps = [int(args.epochs*1/2), int(args.epochs*3/4)]
    else:
        if args.lr == 0.01:
            args.steps = [int(args.epochs*3/4)]
        else:
            args.steps = [int(args.epochs*3/8), int(args.epochs*5/8), int(args.epochs*7/8)]
    dir_name = f'{args.arch}-{args.dataset}'
    dir_name += '-clean' if args.clean else f'-{dataset_index}'
    dir_name += f'.{args.seed}-epoch{args.epochs}'
    for step in args.steps:
        dir_name += f'.{step}'
    dir_name += f'-{args.scenario}' if args.scenario != 'scratch' else ''
    if args.subset_size == 1:
        dir_name += '-full'
    else:
        dir_name += f'-{args.subset_sampler}'
        dir_name += f'-{args.subset_size}'
        args.ctime = 0
        if args.drop_mile:
            dir_name += f'-mile'
            args.miles = [10, 15, 20]
            for step in args.miles:
                dir_name += f'.{step}'
        else:
            dir_name += f'-{args.subset_freq}'
        dir_name += f'-equal' if args.equal_num else ''
        dir_name += f'-start-after-{args.drop_after}' if args.drop_after > 0 else ''
        dir_name += f'-stop-after-{args.stop_after}' if args.stop_after < args.epochs else ''
        dir_name += f'-stochastic' if args.greedy == 'StochasticGreedy' else ''
        dir_name += f'-cosine' if args.metric == 'cosine' else ''
    dir_name += f'-noaug' if args.no_aug else ''
    dir_name += f'-ap' if args.ap_model else ''
    args.out = os.path.join(args.out, dir_name)

    if not os.path.exists(os.path.join(args.out, 'checkpoint.pth.tar')):
        os.makedirs(args.out, exist_ok=True)
    else:
        os.makedirs(args.out)

    # write and save training log
    logging.basicConfig(
        filename=f"{args.out}/output.log",
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger.warning(
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}")

    logger.info(dict(args._get_kwargs()))

    if args.dataset == 'cifar10':
        args.num_classes = 10
        base_dataset = datasets.CIFAR10('./data', train=True, download=True)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_val, download=False)
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        base_dataset = TinyImageNet('./data/tiny-imagenet-200')
        test_dataset = TinyImageNet('./data/tiny-imagenet-200', split='val', transform=transform_val)
    else:
        raise NotImplementedError('Only CIFAR-10 and TinyImageNet are supported.')

    args.train_size = len(base_dataset)
    
    if args.clean:
        train_dataset = PoisonedDataset(
            trainset=base_dataset, 
            indices=np.array(range(len(base_dataset))), 
            transform=transform_train,
            return_index=True,
            size=args.train_size)
        poison_indices = []
        poison_tuples = []
    elif args.poisons_path is not None:
        # load the poisons and their indices within the training set from pickled files
        if os.path.isfile(args.poisons_path):
            with open(args.poisons_path, "rb") as handle:
                print(f"Loading MetaPoison datasets...")
                if args.no_aug:
                    transform_train = transform_val
                poison_data = pickle.load(handle)
                to_pil = transforms.ToPILImage()
                base_dataset.data = np.uint8(poison_data['xtrain'])
                base_dataset.targets = poison_data['ytrain']
                target_img = transform_val(to_pil(np.uint8(poison_data['xtarget'][0])))
                target_class = poison_data['ytarget'][0]
                poisoned_label = poison_data['ytargetadv'][0]
                poison_indices = np.array(range(5000*poisoned_label, 5000*poisoned_label+500))
                poison_tuples = []
                for i in poison_indices:
                    poison_tuples.append((to_pil(np.uint8(poison_data['xtrain'][i])), poison_data['ytrain'][i]))
        else:
            with open(os.path.join(args.poisons_path, "poisons.pickle"), "rb") as handle:
                poison_tuples = pickle.load(handle)
                logger.info(f"{len(poison_tuples)} poisons in this trial.")
                poisoned_label = poison_tuples[0][1]
            with open(os.path.join(args.poisons_path, "base_indices.pickle"), "rb") as handle:
                poison_indices = pickle.load(handle)
            target_img, target_class = get_target(args, transform_val)
        train_dataset = PoisonedDataset(
            trainset=base_dataset, 
            indices=np.array(range(len(base_dataset))), 
            poison_instances=poison_tuples, 
            poison_indices=poison_indices,
            transform=transform_train,
            return_index=True,
            size=args.train_size)
    else:
        raise ValueError('poisons path cannot be empty')

    model = create_model(args)
    model.to(args.device)

    train_val_dataset = PoisonedDataset(
        trainset=base_dataset, 
        indices=np.array(range(len(base_dataset))), 
        poison_instances=poison_tuples, 
        poison_indices=poison_indices,
        transform=transform_val,
        return_index=True,
        size=args.train_size)
    logger.info(f"Target class {target_class}; Poisoned label: {poisoned_label}")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.scenario != 'scratch':
        if args.scenario == 'transfer':
            logger.info("==> Resuming from checkpoint..")
            assert os.path.isfile(
                args.resume), "Error: no checkpoint directory found!"
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("==> Freezing the feature representation..")
            for param in model.parameters():
                param.requires_grad = False
        else:
            checkpoint = torch.load(os.path.join(args.poisons_path, 'model_state.pth.tar'), map_location='cpu')
            model.load_state_dict(checkpoint)
            model.linear = model.fc
            logger.info("==> Decreasing the learning rate for fine-tuning..")
            args.lr = 1e-4
        logger.info("==> Reinitializing the classifier..")
        num_ftrs = model.linear.in_features
        model.linear = nn.Linear(num_ftrs, args.num_classes).to(args.device)  # requires grad by default

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steps)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    args.start_epoch = 0
    
    if args.resume and (args.scenario == 'scratch'):
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    subset = np.array(range(len(base_dataset)))

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")

    model.zero_grad()
    model_to_save = model.module if hasattr(model, "module") else model
    save_checkpoint({
        'epoch': 0,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, is_best=False, checkpoint=args.out, filename='init.pth.tar')

    train(args, train_loader, test_loader, model, optimizer, scheduler, target_img, target_class, poisoned_label, train_val_dataset, train_dataset, loss_fn, poison_indices, subset, base_dataset, poison_tuples)


def train(args, trainloader, test_loader, model, optimizer, scheduler, target_img, target_class, poisoned_label, train_val_dataset, train_dataset, loss_fn, poison_indices, subset, base_dataset, poison_tuples):
    global best_acc
    test_accs = []
    end = time.time()

    model.train()
    N = args.train_size
    B = int(args.subset_size * args.train_size)
    times_selected = torch.zeros(N)

    train_val_loader = DataLoader(
        train_val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"***** Epoch {epoch} *****")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        num_poison_selected = torch.tensor(0)

        if (args.subset_size < 1) and (((epoch % args.subset_freq == 0) and epoch >= args.drop_after and epoch < args.stop_after and epoch <= args.steps[-1] and not args.drop_mile) or (args.drop_mile and epoch in args.miles)):
            logger.info(f'Identifying small clusters at epoch {epoch}...')
            if args.subset_sampler == 'rand':
                subset = get_random_subset(B, N)
            else:
                subset = get_subset(args, model, train_val_loader, B, epoch, N, train_dataset.indices)
            keep = np.where(times_selected[subset] == epoch)[0]
            subset = subset[keep]
            pruned_dataset = PoisonedDataset(
                trainset=base_dataset, 
                indices=subset, 
                poison_instances=poison_tuples, 
                poison_indices=poison_indices,
                transform=transform_train,
                return_index=True,
                size=args.train_size)
            trainloader = DataLoader(
                pruned_dataset,
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                drop_last=False)

        args.eval_step = len(trainloader)
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        model.train()
        
        for batch_idx, batch_input in enumerate(trainloader):
            input, targets, p, index = batch_input
            targets = targets.long()
            num_poison_selected += torch.sum(p)

            data_time.update(time.time() - end)
            logits = model(input.to(args.device))

            loss = loss_fn(logits, targets.to(args.device)).mean()
            loss.backward()

            losses.update(loss.item())
            optimizer.step()
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg))
                p_bar.update()

            times_selected[index] += 1

        if not args.no_progress:
            p_bar.close()

        test_model = model
        test_loss, test_acc = test(args, test_loader, test_model, loss_fn)

        # test poisoning success
        test_model.eval()
        if args.backdoor:
            target_confs= []
            target_preds =[]
            p_accs = []
            t_accs = []
            for t in target_img:
                target_conf = torch.softmax(test_model(t.unsqueeze(0).to(args.device)), dim=-1)
                target_pred = target_conf.max(1)[1].item()
                p_acc = (target_pred == poisoned_label)
                t_acc = (target_pred == target_class)

                target_confs.append(target_conf)
                target_preds.append(target_pred)
                p_accs.append(p_acc)
                t_accs.append(t_acc)

            p_acc = np.mean(p_accs)
            t_acc = np.mean(t_accs)
        else:
            target_conf = torch.softmax(test_model(target_img.unsqueeze(0).to(args.device)), dim=-1)
            target_pred = target_conf.max(1)[1].item()
            p_acc = (target_pred == poisoned_label)
            t_acc = (target_pred == target_class)

        print(f"Poison acc: {p_acc}")
        scheduler.step()

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
            best_p_acc = p_acc

        model_to_save = model.module if hasattr(model, "module") else model
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'test_acc': test_acc,
            'test_loss': test_loss,
            'poison_acc': p_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'times_selected': times_selected,
        }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}'.format(
            np.mean(test_accs[-20:])))
        logger.info(f'Target acc: {t_acc}')
        logger.info(f'Poison acc: {p_acc}\n')


def get_subset(args, model, trainloader, num_sampled, epoch, N, indices):
    if not args.no_progress:
        trainloader = tqdm(trainloader)

    num_classes = model.linear.weight.shape[0]

    grad_preds = []
    labels = []
    conf_all = np.zeros(N)
    conf_true = np.zeros(N)

    with torch.no_grad():
        for _, (inputs, targets, _, index) in enumerate(trainloader):
            model.eval()
            targets = targets.long()

            inputs = inputs.to(args.device)
            if args.backdoor:
                confs = torch.softmax(model(inputs), dim=1).cpu().detach()
                conf_all[index] = np.amax(confs.numpy(), axis=1)
                conf_true[index] = confs[range(len(targets)), targets].numpy()
                g0 = confs - torch.eye(num_classes)[targets.long()]
                grad_preds.append(g0.cpu().detach().numpy())
            else:
                embed = model(inputs, penu=True)
                confs = torch.softmax(torch.matmul(embed, model.linear.weight.T), dim=1).cpu().detach()
                conf_all[index] = np.amax(confs.numpy(), axis=1)
                conf_true[index] = confs[range(len(targets)), targets].numpy()
                embed = embed.cpu().detach()
                g0 = confs - torch.eye(num_classes)[targets.long()]
                grad_preds.append(g0.cpu().detach().numpy())
            targets = targets.numpy()
            labels.append(targets)
        
        labels = np.concatenate(labels)
        subset, subset_weights, _, _, cluster_ = get_coreset(np.concatenate(grad_preds), labels, len(labels), num_sampled, num_classes, equal_num=args.equal_num, optimizer=args.greedy, metric=args.metric)

    subset = indices[subset]
    cluster = -np.ones(N, dtype=int)
    cluster[indices] = cluster_

    keep_indices = np.where(subset_weights > args.cluster_thresh)
    if epoch >= args.drop_after:
        keep_indices = np.where(np.isin(cluster, keep_indices))[0]
        subset = keep_indices
    else:
        subset = np.arange(N)

    return subset


def test(args, test_loader, model, loss_fn):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets).mean()

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
