# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Train a  model on a custom dataset
    这个文件是yolov3的训练脚本。
    抓住 数据 + 模型 + 学习率 + 优化器 + 训练这五步即可。
    Train a YOLOv3 model on a custom dataset.
"""
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, NCOLS, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # 这个 Worker 是这台机器上的第几个 Worker
RANK = int(os.getenv('RANK', -1))  # 这个 Worker 是全局第几个 Worker
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # 总共有几个 Worker


def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks
          ):
    """
        :params hyp: data/hyps/hyp.scratch.yaml   hyp dictionary
        :params opt: main中opt参数
        :params device: 当前设备
    """
    # ----------------------------------------------- 初始化参数和配置信息 ----------------------------------------------
    # 初始化pt参数 + 路径信息 + 超参设置保存 + 保存opt + 加载数据配置信息 + 打印日志信息(logger + wandb) + 其他参数(plots、cuda、nc、names、is_coco)
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # Directories
    w = save_dir / 'weights'  # 保存权重的路径 如runs/train/exp18/weights
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters超参数
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 加载hyp超参信息
    # 日志输出超参信息 hyperparameters: ...
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    with open(save_dir / 'hyp.yaml', 'w') as f:  # 保存hyp
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:  # 保存opt
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # 日志记录器
    if RANK in [-1, 0]:  # 仅在主要进程中初始化日志记录器
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # 创建日志记录器实例
        if loggers.wandb:  # 如果使用 wandb 进行日志记录
            data_dict = loggers.wandb.data_dict
            if resume:  # 如果是恢复训练
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # 从 opt 中获取权重、epochs 和超参数

        # 注册回调函数
        for k in methods(loggers):  # 获取 loggers 的方法
            callbacks.register_action(k, callback=getattr(loggers, k))  # 注册每个方法为回调函数

    # 配置
    plots = not evolve  # 创建绘图
    cuda = device.type != 'cpu'  # 检查是否使用 GPU
    init_seeds(1 + RANK)  # 初始化随机种子
    with torch_distributed_zero_first(LOCAL_RANK):  # 在分布式训练中确保只有一个进程执行
        data_dict = data_dict or check_dataset(data)  # 检查数据集，如果为 None 则验证数据集
    train_path, val_path = data_dict['train'], data_dict['val']  # 获取训练和验证路径
    nc = 1 if single_cls else int(data_dict['nc'])  # 类别数量
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # 类别名称
    # 检查类别名称与类别数量是否匹配
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # 检查是否为 COCO 数据集

    # Model
    check_suffix(weights, '.pt')  # 检查权重文件后缀
    pretrained = weights.endswith('.pt')  # 判断是否为预训练模型
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):  # 在分布式训练中确保只有一个进程执行下载
            weights = attempt_download(weights)  # 如果本地不存在，则下载权重文件
        ckpt = torch.load(weights, map_location=device)  # 加载检查点
        # 这里加载模型有两种方式，一种是通过opt.cfg 另一种是通过ckpt['model'].yaml
        # 区别在于是否使用resume 如果使用resume会将opt.cfg设为空，按照ckpt['model'].yaml来创建模型
        # 这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
        # 原因: 保存的模型会保存anchors，有时候用户自定义了anchor之后，再resume，则原来基于coco数据集的anchor会自己覆盖自己设定的anchor
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建模型
        # 排除的键
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []
        csd = ckpt['model'].float().state_dict()  # 将检查点的状态字典转换为 FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # 取交集
        model.load_state_dict(csd, strict=False)  # 加载状态字典
        LOGGER.info(f'从 {weights} 转移了 {len(csd)}/{len(model.state_dict())} 项')  # 记录转移的项目数量
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建模型

    # 冻结层
    freeze = [f'model.{x}.' for x in range(freeze)]  # 要冻结的层
    for k, v in model.named_parameters():
        v.requires_grad = True  # 允许所有层进行训练
        if any(x in k for x in freeze):  # 如果当前参数名在冻结列表中
            LOGGER.info(f'冻结 {k}')  # 记录冻结的层
            v.requires_grad = False  # 取消该层的梯度计算

    # Image size
    gs = max(int(model.stride.max()), 32)  # 网格大小（最大步幅）
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # 验证图像大小是 gs 的倍数

    # Batch size
    if RANK == -1 and batch_size == -1:  # 仅在单 GPU 下，估计最佳批量大小
        batch_size = check_train_batch_size(model, imgsz)  # 检查并确定最佳批量大小

    # Optimizer
    nbs = 64  # 名义批量大小
    accumulate = max(round(nbs / batch_size), 1)  # 在优化前累积损失
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # 根据批量大小缩放权重衰减
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # 优化器参数组
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # 偏置
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # 权重（不衰减）
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # 权重（有衰减）
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # 调整 beta1 为动量
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # 添加 g1（带衰减的权重）
    optimizer.add_param_group({'params': g2})  # 添加 g2（偏置）

    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # 学习率调度器
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # 线性学习率调整
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # 余弦学习率调整，从 1 到 hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 创建学习率调度器
    # plot_lr_scheduler(optimizer, scheduler, epochs)  # 可选：绘制学习率调度曲线

    # 单卡训练: 使用EMA（指数移动平均）对模型的参数做平均, 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒。
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # 使用预训练
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # 优化器
        if ckpt['optimizer'] is not None:  # 如果存在优化器状态
            optimizer.load_state_dict(ckpt['optimizer'])  # 加载优化器状态
            best_fitness = ckpt['best_fitness']  # 更新最佳性能

        # EMA（指数移动平均）
        if ema and ckpt.get('ema'):  # 如果启用了 EMA
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # 加载 EMA 状态
            ema.updates = ckpt['updates']  # 更新次数

        # 训练轮数
        start_epoch = ckpt['epoch'] + 1  # 从检查点获取起始轮数
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:  # 假设你设定的epoch为100次，但是此时加载的模型已经训练150次，则接着再训练100次。如果小于则训练剩余的部分。
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # 是否使用DP mode
    # 如果rank=-1且gpu数量>1，则使用DataParallel单机多卡模式，效果并不好（分布不平均）
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm  是否使用跨卡BN
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    # 创建训练数据加载器和数据集
    train_loader, dataset = create_dataloader(
        train_path,  # 训练数据路径
        imgsz,  # 图像大小
        batch_size // WORLD_SIZE,  # 每个进程的批大小
        gs,  # 网格大小
        single_cls,  # 是否为单类检测
        hyp=hyp,  # 超参数
        augment=True,  # 是否进行数据增强
        cache=opt.cache,  # 是否缓存数据
        rect=opt.rect,  # 是否使用矩形批次
        rank=LOCAL_RANK,  # 当前进程的rank
        workers=workers,  # 数据加载的工作线程数量
        image_weights=opt.image_weights,  # 是否使用图像权重
        quad=opt.quad,  # 是否使用四元组格式
        prefix=colorstr('train: '),  # 前缀，用于输出信息
        shuffle=True  # 是否打乱数据
    )
    # 获取最大标签类
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # 批次数量
    # 断言标签类不超过类别数
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        # 创建验证数据加载器
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            # 合并标签
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # 类别
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # 频率
            # model._initialize_biases(cf.to(device))

            if plots:
                # 绘制标签分布图
                plot_labels(labels, names, save_dir)

            # 检查锚框
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # 预减锚框精度

        # 运行回调函数
        callbacks.run('on_pretrain_routine_end')

    # DDP模式
    if cuda and RANK != -1:
        # 将模型包装为分布式数据并行模式
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    nl = de_parallel(model).model[-1].nl  # 获取检测层的数量 (number of detection layers)
    hyp['box'] *= 3 / nl  # 将框的损失缩放到检测层数
    hyp['cls'] *= nc / 80 * 3 / nl  # 将类别损失缩放到类别数量和检测层数
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # 将物体损失缩放到图像大小和检测层数
    hyp['label_smoothing'] = opt.label_smoothing  # 设置标签平滑参数
    model.nc = nc  # 将类别数量附加到模型
    model.hyp = hyp  # 将超参数附加到模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 计算并附加类别权重
    model.names = names  # 将类名附加到模型

    # Start training
    t0 = time.time()  # 记录开始时间
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # 计算暖身迭代次数，至少1000次
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # 限制暖身迭代次数为训练的一半
    last_opt_step = -1  # 最后优化步数
    maps = np.zeros(nc)  # 每个类别的mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # 初始结果，包含P, R, mAP@.5, mAP@.5-.95, val_loss
    scheduler.last_epoch = start_epoch - 1  # 不移动调度器的最后一个epoch
    scaler = amp.GradScaler(enabled=cuda)  # 初始化混合精度训练的Scaler
    stopper = EarlyStopping(patience=opt.patience)  # 初始化早停类
    compute_loss = ComputeLoss(model)  # 初始化损失计算类
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # 开始训练循环，遍历每个epoch
        model.train()  # 设置模型为训练模式

        # 可选：更新图像权重（仅限单GPU）
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # 计算类别权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # 计算图像权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # 随机选择加权的索引

        mloss = torch.zeros(3, device=device)  # 初始化平均损失
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)  # 设置DistributedSampler的epoch
        pbar = enumerate(train_loader)  # 获取训练数据的迭代器
        print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # 显示进度条

        optimizer.zero_grad()  # 重置梯度
        for i, (
        imgs, targets, paths, _) in pbar:  # 遍历每个批次 -------------------------------------------------------------
            ni = i + nb * epoch  # 计算自训练开始以来的累计批次数
            imgs = imgs.to(device, non_blocking=True).float() / 255  # 将图像转换为浮点数并归一化

            # 热身阶段
            if ni <= nw:
                xi = [0, nw]  # x插值
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())  # 计算累计的批次数
                for j, x in enumerate(optimizer.param_groups):
                    # 对每个参数组设置学习率和动量
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # 多尺度训练
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # 随机选择图像大小
                sf = sz / max(imgs.shape[2:])  # 计算缩放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # 计算新形状
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)  # 调整图像大小

            # 前向传播
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # 模型前向传播
                loss, loss_items = compute_loss(pred, targets.to(device))  # 计算损失
                if RANK != -1:
                    loss *= WORLD_SIZE  # 在DDP模式下缩放损失

            # 反向传播
            scaler.scale(loss).backward()  # 反向传播计算梯度

            # 优化
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # 更新优化器
                scaler.update()  # 更新缩放器
                optimizer.zero_grad()  # 重置梯度
                if ema:
                    ema.update(model)  # 更新指数移动平均
                last_opt_step = ni  # 更新最后优化步骤

            # 日志记录
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # 显示内存使用情况
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

        # 学习率调度
        lr = [x['lr'] for x in optimizer.param_groups]  # 记录学习率
        scheduler.step()  # 更新学习率

        if RANK in [-1, 0]:  # 如果是主进程（或单GPU）
            # 计算mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)  # 调用回调函数，标记epoch结束
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])  # 更新EMA模型的属性
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop  # 判断是否为最后一个epoch或是否可以停止
            if not noval or final_epoch:  # 如果需要验证或是最后一个epoch
                results, maps, _ = val.run(data_dict,  # 进行验证
                                           batch_size=batch_size // WORLD_SIZE * 2,  # 设置批大小
                                           imgsz=imgsz,  # 输入图像大小
                                           model=ema.ema,  # 使用EMA模型进行验证
                                           single_cls=single_cls,  # 单类检测标志
                                           dataloader=val_loader,  # 验证数据加载器
                                           save_dir=save_dir,  # 保存目录
                                           plots=False,  # 不绘制图表
                                           callbacks=callbacks,  # 回调函数
                                           compute_loss=compute_loss)  # 计算损失

            # 更新最佳mAP
            fi = fitness(np.array(results).reshape(1, -1))  # 计算加权的指标组合 [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:  # 如果当前结果优于历史最佳
                best_fitness = fi  # 更新最佳适应度
            log_vals = list(mloss) + list(results) + lr  # 记录损失和结果
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)  # 调用回调函数，记录epoch结束

            # 保存模型
            if (not nosave) or (final_epoch and not evolve):  # 如果需要保存模型
                ckpt = {'epoch': epoch,  # 记录当前epoch
                        'best_fitness': best_fitness,  # 记录最佳适应度
                        'model': deepcopy(de_parallel(model)).half(),  # 深拷贝模型并转换为半精度
                        'ema': deepcopy(ema.ema).half(),  # 深拷贝EMA模型并转换为半精度
                        'updates': ema.updates,  # EMA更新次数
                        'optimizer': optimizer.state_dict(),  # 优化器状态
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,  # 如果使用wandb，记录ID
                        'date': datetime.now().isoformat()}  # 当前日期时间

                # 保存最新模型和最佳模型，并删除不需要的检查点
                torch.save(ckpt, last)  # 保存最新模型
                if best_fitness == fi:  # 如果当前为最佳适应度
                    torch.save(ckpt, best)  # 保存最佳模型
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):  # 根据保存周期保存模型
                    torch.save(ckpt, w / f'epoch{epoch}.pt')  # 保存当前epoch模型
                del ckpt  # 删除检查点数据以释放内存
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)  # 调用模型保存的回调函数

            # 单GPU模式下的提前停止
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):  # 如果是单GPU并且触发了停止条件
                break  # 退出训练

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:  # 如果是主进程（或单GPU）
        LOGGER.info(
            f'\n{epoch - start_epoch + 1} 个epoch已完成，耗时 {(time.time() - t0) / 3600:.3f} 小时。')  # 记录已完成的epoch和耗时
        for f in last, best:  # 遍历最新模型和最佳模型
            if f.exists():  # 如果模型文件存在
                strip_optimizer(f)  # 去除优化器信息
                if f is best:  # 如果是最佳模型
                    LOGGER.info(f'\n正在验证 {f}...')  # 记录验证信息
                    results, _, _ = val.run(data_dict,  # 进行验证
                                            batch_size=batch_size // WORLD_SIZE * 2,  # 设置批大小
                                            imgsz=imgsz,  # 输入图像大小
                                            model=attempt_load(f, device).half(),  # 加载模型并转换为半精度
                                            iou_thres=0.65 if is_coco else 0.60,  # COCO数据集最佳IOU阈值
                                            single_cls=single_cls,  # 单类检测标志
                                            dataloader=val_loader,  # 验证数据加载器
                                            save_dir=save_dir,  # 保存目录
                                            save_json=is_coco,  # 如果是COCO，保存JSON
                                            verbose=True,  # 显示详细信息
                                            plots=True,  # 绘制图表
                                            callbacks=callbacks,  # 回调函数
                                            compute_loss=compute_loss)  # 计算损失

                    if is_coco:  # 如果是COCO数据集
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness,
                                      fi)  # 调用回调函数，记录信息

        callbacks.run('on_train_end', last, best, plots, epoch, results)  # 训练结束时调用回调
        # LOGGER.info(f"结果已保存到 {colorstr('bold', save_dir)}")  # 记录结果保存路径
        print(f"结果已保存到 {colorstr('bold', save_dir)}")  # 记录结果保存路径

    torch.cuda.empty_cache()  # 清空CUDA缓存
    return results


def parse_opt(known=False):
    """
        函数功能：设置opt参数
    """
    parser = argparse.ArgumentParser()
    # --------------------------------------------------- 常用参数 ---------------------------------------------
    parser.add_argument('--weights', type=str, default=ROOT / 'weight/yolov3.pt', help='initial weights path') # weights: 权重文件
    parser.add_argument('--cfg', type=str, default='models/yolov3.yaml', help='model.yaml path')  # cfg: 网络模型配置文件 包括nc、depth_multiple、width_multiple、anchors、backbone、head等
    parser.add_argument('--data', type=str, default=ROOT / 'data/you.yaml', help='dataset.yaml path') # data: 实现数据集配置文件 包括path、train、val、test、nc、names等
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path') # hyp: 训练时的超参文件
    parser.add_argument('--epochs', type=int, default=50)  # epochs: 训练轮次
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch') # batch-size: 训练批次大小
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=416, help='train, val image size (pixels)') # imgsz: 输入网络的图片分辨率大小
    parser.add_argument('--rect', action='store_true', default=True, help='rectangular training')  # rect: 是否采用Rectangular training/inference，一张图片为长方形，我们在将其送入模型前需要将其resize到要求的尺寸，所以我们需要通过补灰padding来变为正方形的图。
    parser.add_argument('--resume', nargs='?', const=True, default="", help='resume most recent training') # resume: 断点续训, 从上次打断的训练结果处接着训练  默认False
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')  # nosave: 不保存模型  默认保存  store_true: only test final epoch
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')  # noval: 只在最后一次进行测试，默认False
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')  # noautoanchor: 不自动调整anchor 默认False(自动调整anchor)
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations') # evolve: 是否进行超参进化，使得数值更好 默认False
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')   # bucket: 谷歌云盘bucket 一般用不到
    parser.add_argument('--cache', type=str, nargs='?', const='ram', default=True, help='--cache images in "ram" (default) or "disk"')  # cache:是否提前缓存图片到内存，以加快训练速度
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')   #  image-weights: 对于那些训练不好的图片，会在下一轮中增加一些权重
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # device: 训练的设备
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')  # multi-scale: 是否使用多尺度训练 默认False，要被32整除。
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class') # single-cls: 数据集是否只有一个类别 默认False
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer') # adam: 是否使用adam优化器
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode') # sync-bn: 是否使用跨卡同步bn操作,再DDP中使用  默认False
    parser.add_argument('--workers', type=int, default=1, help='max dataloader workers (per RANK in DDP mode)')  # workers: dataloader中的最大work数（线程个数）
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name') # project: 训练结果保存的根目录 默认是runs/train
    parser.add_argument('--name', default='exp', help='save to project/name')  # name: 训练结果保存的目录 默认是exp
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')  # exist_ok: 是否重新创建日志文件, False时重新创建文件(默认文件都是不存在的)
    parser.add_argument('--quad', action='store_true', help='quad dataloader')  # quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
    parser.add_argument('--linear-lr', action='store_true', help='linear LR') # linear-lr：用于对学习速率进行调整，默认为 False，（通过余弦函数来降低学习率）
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')  # label-smoothing: 标签平滑增强 默认0.0不增强  要增强一般就设为0.1
    parser.add_argument('--patience', type=int, default=1000, help='EarlyStopping patience (epochs without improvement)')  # 早停机制，训练到一定的epoch，如果模型效果未提升，就让模型提前停止训练。
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')  # freeze: 使用预训练模型的规定固定权重不进行调整  --freeze 10  :意思从第0层到到第10层不训练
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)') # 设置多少个epoch保存一次模型
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify') # local_rank: rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式

    # --------------------------------------------------- W&B(wandb)参数 ---------------------------------------------
    parser.add_argument('--entity', default=None, help='W&B: Entity') #wandb entity 默认None
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')  # 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval') # 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


def main(opt, callbacks=Callbacks()):
    # 1、logging和wandb初始化
    # 日志初始化
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_git_status()
        check_requirements(exclude=['thop'])

    # 2、使用断点续训 就从last.pt中读取相关参数；不使用断点续训 就从文件中读取相关参数
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        # 使用断点续训 就从last.pt中读取相关参数
        # 如果resume是str，则表示传入的是模型的路径地址
        # 如果resume是True，则通过get_lastest_run()函数找到runs为文件夹中最近的权重文件last.pt
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'

        # 相关的opt参数也要替换成last.pt中的opt参数
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        # 不使用断点续训 就从文件中读取相关参数
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        # 根据opt.project生成目录  如: runs/train/exp18
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # 3、DDP mode设置
    # 选择设备  cpu/cuda:0
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        # LOCAL_RANK != -1 进行多GPU训练
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        # 根据GPU编号选择设备
        device = torch.device('cuda', LOCAL_RANK)
        # 初始化进程组  distributed backend
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # 4、不使用进化算法 正常Train
    if not opt.evolve:
        # 如果不进行超参进化 那么就直接调用train()函数，开始训练
        train(opt.hyp, opt, device, callbacks)
        # 如果是使用多卡训练, 那么销毁进程组
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # 5、遗传进化算法，边进化边训练
    # 否则使用超参进化算法(遗传算法) 求出最佳超参 再进行训练
    else:
        # 超参进化列表 (突变规模, 最小值, 最大值)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # 初始学习率 (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # 最终 OneCycleLR 学习率 (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD 动量/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # 优化器权重衰减
            'warmup_epochs': (1, 0.0, 5.0),  # 预热轮数 (可为小数)
            'warmup_momentum': (1, 0.0, 0.95),  # 预热初始动量
            'warmup_bias_lr': (1, 0.0, 0.2),  # 预热初始偏置学习率
            'box': (1, 0.02, 0.2),  # 框损失增益
            'cls': (1, 0.2, 4.0),  # 分类损失增益
            'cls_pw': (1, 0.5, 2.0),  # 分类 BCELoss 正权重
            'obj': (1, 0.2, 4.0),  # 目标损失增益 (根据像素缩放)
            'obj_pw': (1, 0.5, 2.0),  # 目标 BCELoss 正权重
            'iou_t': (0, 0.1, 0.7),  # IoU 训练阈值
            'anchor_t': (1, 2.0, 8.0),  # 锚框倍数阈值
            'anchors': (2, 2.0, 10.0),  # 每个输出网格的锚框数量 (0 表示忽略)
            'fl_gamma': (0, 0.0, 2.0),  # 焦点损失 gamma (efficientDet 默认 gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # 图像 HSV-色调 增强 (比例)
            'hsv_s': (1, 0.0, 0.9),  # 图像 HSV-饱和度 增强 (比例)
            'hsv_v': (1, 0.0, 0.9),  # 图像 HSV-明度 增强 (比例)
            'degrees': (1, 0.0, 45.0),  # 图像旋转 (+/- 角度)
            'translate': (1, 0.0, 0.9),  # 图像平移 (+/- 比例)
            'scale': (1, 0.0, 0.9),  # 图像缩放 (+/- 增益)
            'shear': (1, 0.0, 10.0),  # 图像剪切 (+/- 角度)
            'perspective': (0, 0.0, 0.001),  # 图像透视 (+/- 比例), 范围 0-0.001
            'flipud': (1, 0.0, 1.0),  # 图像上下翻转 (概率)
            'fliplr': (0, 0.0, 1.0),  # 图像左右翻转 (概率)
            'mosaic': (1, 0.0, 1.0),  # 图像混合 (概率)
            'mixup': (1, 0.0, 1.0),  # 图像混合 (概率)
            'copy_paste': (1, 0.0, 1.0)  # 分割复制粘贴 (概率)
        }
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 载入初始超参
            if 'anchors' not in hyp:  # 如果 hyp.yaml 中没有定义 anchors
                hyp['anchors'] = 3  # 设置默认值为 3
        # 设置 opt.noval 和 opt.nosave 为 True，并指定保存目录
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # 只在最终 epoch 验证/保存

        # evolvable indices 代码被注释掉了
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # 可进化的索引
        # 超参进化后文件保存地址
        evolve_yaml = save_dir / 'hyp_evolve.yaml'
        evolve_csv = save_dir / 'evolve.csv'
        # 如果指定了云存储桶，下载 evolve.csv 文件
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # 如果存在，下载 evolve.csv
        """
           使用遗传算法进行参数进化 默认是进化300代
           这里的进化算法是：根据之前训练时的hyp来确定一个base hyp再进行突变；
           如何根据？通过之前每次进化得到的results来确定之前每个hyp的权重
           有了每个hyp和每个hyp的权重之后有两种进化方式；
               1.根据每个hyp的权重随机选择一个之前的hyp作为base hyp，random.choices(range(n), weights=w)
               2.根据每个hyp的权重对之前所有的hyp进行融合获得一个base hyp，(x * w.reshape(n, 1)).sum(0) / w.sum()
           evolve.txt会记录每次进化之后的results+hyp
           每次进化时，hyp会根据之前的results进行从大到小的排序，再根据fitness函数计算之前每次进化得到的hyp的权重，再确定哪一种进化方式，从而进行进化。
        """
        for _ in range(opt.evolve):  # 迭代进行超参数进化
            if evolve_csv.exists():  # 如果 evolve.csv 存在：选择最佳超参并进行变异
                # 选择父代
                parent = 'single'  # 父代选择方法：'single' 或 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # 考虑的之前结果数量
                x = x[np.argsort(-fitness(x))][:n]  # 前 n 个变异结果
                w = fitness(x) - fitness(x).min() + 1E-6  # 权重 (总和 > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # 随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  # 加权选择
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合
                # 变异
                mp, s = 0.8, 0.2  # 变异概率, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # 增益 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # 变异直到发生变化 (防止重复)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # 变异
            # 限制在范围内
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # 下限
                hyp[k] = min(hyp[k], v[2])  # 上限
                hyp[k] = round(hyp[k], 5)  # 有效数字
            # 训练变异后的超参
            results = train(hyp.copy(), opt, device, callbacks)
            # 写入变异结果
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)
        # 绘制结果
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov3.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
