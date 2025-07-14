import argparse
from copy import deepcopy
import logging
import os
import pprint
import sys
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from Depth.depth_anything_v2.dpt import DepthAnythingV2
import torch.nn.functional as F
from util.consistency import consistency_weight
import time
from DepthV1.depth_anything.dpt import DepthAnything

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', type=str, default='/configs/pascal.yaml')
parser.add_argument('--labeled-id-path', type=str, default='/splits/pascal/366/labeled.txt')
parser.add_argument('--unlabeled-id-path', type=str, default='/splits/pascal/366/unlabeled.txt')
parser.add_argument('--save-path', type=str, default='/VOC_Results')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def depth_difference_consistency_loss_global(A, D):
    """
    计算深度差异与语义特征差异的对齐损失
    Args:
        A: 语义分割特征图 (B, C, H, W)
        D: 深度图 (B, 1, H, W)
    Returns:
        loss: 深度差异一致性损失
    """
    B, C, H, W = A.shape
    D = F.interpolate(D.unsqueeze(1), (H//8, W//8), mode="bilinear", align_corners=True)
    A = F.interpolate(A, (H//8, W//8), mode="bilinear", align_corners=True)
    # 将深度图展平为 (B, H*W)
    D_flat = D.view(B, -1) # [4, 21904]
    A_flat = A.view(B, C, -1) # [4, 64, 21904]


    # 计算深度差异矩阵，(B, H*W, H*W)
    depth_diff_matrix = torch.abs(D_flat.unsqueeze(2) - D_flat.unsqueeze(1))

    # 计算语义特征差异矩阵，(B, H*W, H*W)
    A_diff = A_flat.unsqueeze(3) - A_flat.unsqueeze(2)  # (B, C, H*W, H*W)
    A_diff_matrix = torch.norm(A_diff, dim=1)
    # print(depth_diff_matrix.size(), A_diff_matrix.size())


    # 最小化深度差异与语义差异之间的差异 (可以用 L2 或者 L1 损失)
    loss = F.mse_loss(A_diff_matrix, depth_diff_matrix) / (H//8 * W//8) 
    
    return loss


def depth_difference_consistency_loss(A, D,conf,mask,ignore_mask,rank):
    """
    计算深度差异与语义特征差异的对齐损失
    Args:
        A: 语义分割特征图 (B, C, H, W)
        D: 深度图 (B, H, W)
    Returns:
        loss: 深度差异一致性损失
    """
    B, C, H, W = A.shape
    D = F.interpolate(D.unsqueeze(1), (H//8, W//8), mode="bilinear", align_corners=True)
    ignore_mask = F.interpolate(ignore_mask.unsqueeze(1).float(), (H//8, W//8), mode="bilinear", align_corners=True).squeeze(1)
    A = F.interpolate(A, (H//8, W//8), mode="bilinear", align_corners=True)
    mask = F.interpolate(mask.unsqueeze(1).float(), (H//8, W//8), mode="bilinear", align_corners=True).squeeze(1).long()
    conf = F.interpolate(conf.float(), (H//8, W//8), mode="bilinear", align_corners=True) # B C H W


    log_probs = torch.log(conf + 1e-10)
    absolute_entropy = torch.abs(-torch.sum(conf * log_probs, dim=1)) # 得到对应的绝对熵值


    processed_depth = D.clone()  # 复制深度图，用于修改

    # 获取所有类别
    unique_classes = torch.unique(mask)  # 获取分割图中的所有类别标签
    num_classes = len(unique_classes)

    total_loss  = torch.tensor(0.0,requires_grad=True)  # 初始化损失

    for class_idx in unique_classes:
        # 遍历每个批次的图片
        class_loss = torch.tensor(0.0,requires_grad=True)  # 初始化损失
        class_inter_loss = torch.tensor(0.0,requires_grad=True)  # 初始化损失
        Prev_class = None
        for batch_idx in range(B):
            seg_map = mask[batch_idx]   # (H, W)
            depth = D[batch_idx, 0]  # 提取该批次的深度图 (H, W)
            feature_each_batch = A[batch_idx]
            ignore_m = ignore_mask[batch_idx]

            entropy_each_batch = absolute_entropy[batch_idx]

            # 找到该类别的区域（布尔掩码）
            class_mask = ((seg_map == class_idx) & (ignore_m != 255))

            if class_mask.sum() == 0:  # 如果该类别区域为空，跳过
                continue

            # 获取该类别区域的深度值
            depth_values = depth[class_mask]  # 形状为 (N,), N 是该类别的像素数
            feature_each_class = feature_each_batch[:,class_mask]  # C H  W 
            entropy_each_batch_class = entropy_each_batch[class_mask]


            processed_depth[batch_idx, 0] = depth  ## H W 

            D_flat = depth[class_mask].view(-1)
            depth_diff_matrix = torch.abs(D_flat.unsqueeze(1) - D_flat.unsqueeze(0))  # # (B, C, H*W, H*W)
            weights = torch.exp(-(depth_diff_matrix ** 2) / (2))  # 权重

            A_flat = feature_each_class.view(C,-1)

            A_diff = A_flat.unsqueeze(2) - A_flat.unsqueeze(1)  # ( C, H*W, H*W)
            A_diff_matrix = torch.norm(A_diff, dim=0)

            entropy_diff = entropy_each_batch_class.view(-1)
            entropy_diff_matrix = torch.abs(entropy_diff.unsqueeze(1) - entropy_diff.unsqueeze(0))

            class_loss  = class_loss  + torch.mean((F.mse_loss(A_diff_matrix, weights,reduction='none')  *   entropy_diff_matrix))

            #### 类间损失 ####
            if Prev_class is not None:
                #### 上个类的特征 ####
                feature_pre_flat = Prev_class_feature.view(C,-1)
                depth_difference = torch.abs(D_flat.unsqueeze(1) - prev_class_depth.unsqueeze(0))  ###当前的减去前一个类别 (H*W, H*W)
                feature_difference = A_flat.unsqueeze(2) - feature_pre_flat.unsqueeze(1) 
                feature_difference_matrix = torch.norm(feature_difference, dim=0) # (H*W, H*W)
                calculate_mask = (depth_difference > 1).float() 
                class_inter_loss  = class_inter_loss + torch.mean((F.mse_loss(feature_difference_matrix, depth_difference,reduction='none') * calculate_mask))
 

        Prev_class = class_idx
        Prev_class_feature = feature_each_class
        prev_class_depth = D_flat
        

        total_loss = total_loss + class_loss / B + class_inter_loss / B
 
    return total_loss / (num_classes)




####not use####
def depth_difference_consistency_loss_cos_similarity(A, D,conf,mask,ignore_mask,rank):
    """
    计算深度差异与语义特征差异的对齐损失
    Args:
        A: 语义分割特征图 (B, C, H, W)
        D: 深度图 (B, H, W)
    Returns:
        loss: 深度差异一致性损失
    """
    B, C, H, W = A.shape
    D = F.interpolate(D.unsqueeze(1), (H//8, W//8), mode="bilinear", align_corners=True)
    ignore_mask = F.interpolate(ignore_mask.unsqueeze(1).float(), (H//8, W//8), mode="bilinear", align_corners=True).squeeze(1)
    A = F.interpolate(A, (H//8, W//8), mode="bilinear", align_corners=True)
    mask = F.interpolate(mask.unsqueeze(1).float(), (H//8, W//8), mode="bilinear", align_corners=True).squeeze(1).long()
    conf = F.interpolate(conf.float(), (H//8, W//8), mode="bilinear", align_corners=True) # B C H W

    log_probs = torch.log(conf + 1e-10)
    absolute_entropy = torch.abs(-torch.sum(conf * log_probs, dim=1)) # 得到对应的绝对熵值
    processed_depth = D.clone()  # 复制深度图，用于修改

    # 获取所有类别
    unique_classes = torch.unique(mask)  # 获取分割图中的所有类别标签
    num_classes = len(unique_classes)

    total_loss  = torch.tensor(0.0,requires_grad=True)  # 初始化损失

    for class_idx in unique_classes:
        # 遍历每个批次的图片
        class_loss = torch.tensor(0.0,requires_grad=True)  # 初始化损失
        for batch_idx in range(B):
            seg_map = mask[batch_idx]   # (H, W)
            depth = D[batch_idx, 0]  # 提取该批次的深度图 (H, W)
            feature_each_batch = A[batch_idx]
            ignore_m = ignore_mask[batch_idx]

            entropy_each_batch = absolute_entropy[batch_idx]

            # 找到该类别的区域（布尔掩码）
            class_mask = ((seg_map == class_idx) & (ignore_m != 255))

            if class_mask.sum() == 0:  # 如果该类别区域为空，跳过
                continue

            # 获取该类别区域的深度值
            depth_values = depth[class_mask]  # 形状为 (N,), N 是该类别的像素数
            feature_each_class = feature_each_batch[:,class_mask]  # C H  W 

            feature_each_class = feature_each_class.view(C,-1)
            similarity_matrix = F.cosine_similarity(feature_each_class.unsqueeze(2), feature_each_class.unsqueeze(1), dim=0)

            num_pairs = (H * W) * (H * W - 1) / 2

            mean_similarity = similarity_matrix.triu(diagonal=1).sum() / num_pairs
            class_loss = 1 - mean_similarity


        total_loss = total_loss + class_loss / B

    return total_loss / (num_classes)



def depth_guided_class_specific_consistency_loss(logits, depth_map, label, ignore_mask, threshold=0.95, margin=0.2, ignore_value=255):
    """
    Class-specific depth-guided consistency loss using weakly-augmented predictions (label).
    
    Args:
        logits: Logits from the model (B, num_classes, H, W)
        depth_map: Depth map (B, 1, H, W)
        label: Weakly-augmented prediction labels (B, H, W)
        ignore_mask: Ignore mask for regions that shouldn't be considered in the loss
        threshold: Confidence threshold for defining low-confidence regions
        margin: Tolerance margin for depth difference in consistency loss
        ignore_value: Value in ignore_mask indicating regions to ignore in the loss calculation
    
    Returns:
        loss: Consistency loss value based on class-specific low-confidence regions
    """
    B, num_classes, H, W = logits.shape

    # Downscale depth map and ignore mask to match the resolution of the logits
    depth_map = F.interpolate(depth_map.unsqueeze(1), size=(H // 10, W // 10), mode='bilinear', align_corners=True)
    ignore_mask = F.interpolate(ignore_mask.unsqueeze(1).float(), size=(H // 10, W // 10), mode="bilinear").squeeze(1)

    # Downscale logits and labels for consistency with the reduced resolution
    logits = F.interpolate(logits, size=(H // 10, W // 10), mode='bilinear', align_corners=True)

    label = F.interpolate(label.unsqueeze(1).float(), size=(H // 10, W // 10), mode="bilinear").squeeze(1).long()
    
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Identify unique classes in the current batch
    unique_classes = label.unique()
    total_loss = 0.0
    count = 0

    threshold_depth_difference = 1.5

    indices = torch.randperm(unique_classes.numel())
    unique_classes_shuffle = unique_classes[indices] # 将类别序列打乱

    prev_class_logits = None
    prev_class_depth = None
    pre_class_mask = None
    # Calculate depth differences between each pair of pixels
    depth_flat = depth_map.view(B, -1)  # (B, H*W)
    depth_diff_matrix = torch.abs(depth_flat.unsqueeze(2) - depth_flat.unsqueeze(1))  # (B, H*W, H*W) 每个像素之间的差异都有了
    logits_flat = logits.view(B, num_classes, -1)  # (B, num_classes, H*W)
    for c in unique_classes_shuffle:
        if c == ignore_value:
            continue  # Skip ignore class

        # Mask for pixels predicted as class `c` in the weakly-augmented label
        class_mask = (label == c).float()  # Only pixels where the weak prediction is class `c`

        # Define low-confidence regions within this class, excluding ignored areas
        low_confidence_mask = ((probabilities[:, c, :, :] < threshold) & (ignore_mask != ignore_value)).float()
        class_low_conf_mask = low_confidence_mask * class_mask  # Only low-confidence pixels within class `c`

        if class_low_conf_mask.sum() == 0:
            # Skip if no low-confidence pixels in the current class
            continue

        # Flatten depth map and low-confidence mask for matrix calculations
        class_low_conf_mask_flat = class_low_conf_mask.view(B, -1)  # Flatten (B, H*W)
        
        logits_flat_clone = logits_flat.permute(1,0,2)
        current_low_pixel_logit = logits_flat_clone[:,class_low_conf_mask_flat.type(torch.bool)]  # num_classes * N
        current_low_pixel_reverse = current_low_pixel_logit.permute(1,0)  #  N * num_classes


        # Compute logit similarity matrix for class `c`
        logits_class_flat = logits_flat[:, c, :]  # (B, H*W)
        logit_similarity_matrix = torch.matmul(logits_class_flat.unsqueeze(2), logits_class_flat.unsqueeze(1))  # (B, H*W, H*W)
        logit_similarity_matrix = torch.sigmoid(logit_similarity_matrix)  # Optional: Apply sigmoid to keep values in [0, 1]
        
        # Depth-guided mask for regions with similar depth
        consistency_mask = (depth_diff_matrix < margin).float()  # Mask of pixels with similar depth (B, H*W, H*W)

        consistency_mask_inter = (depth_diff_matrix > threshold_depth_difference).float()  # Mask of pixels with similar depth (B, H*W, H*W)

        
        # Mask for low-confidence regions within class `c` with similar depth
        low_confidence_region_mask = class_low_conf_mask_flat.unsqueeze(2) * class_low_conf_mask_flat.unsqueeze(1)  # (B, H*W, H*W)
        combined_mask = low_confidence_region_mask * consistency_mask
        combined_mask = torch.clamp(combined_mask, min=1e-6)  # Avoid zero values

        if prev_class_logits is not None:
            low_confidence_inter_mask = pre_class_mask.unsqueeze(2) * class_low_conf_mask_flat.unsqueeze(1) # [4, 2601, 2601] 
            mask_inter = low_confidence_inter_mask * consistency_mask_inter # [4, 2601, 2601]
            mask_inter = torch.clamp(mask_inter, min=1e-6) # [4, 2601, 2601]
            logit_simialarity = torch.matmul(prev_class_logits.unsqueeze(2),logits_class_flat.unsqueeze(1))  # [162, 1]
            loss_inter_class = torch.mean(mask_inter * torch.abs(torch.sigmoid(logit_simialarity))) / mask_inter.sum()


        # Calculate consistency loss for class `c`
        class_loss = torch.mean(combined_mask * torch.abs(1 - logit_similarity_matrix))
        
        # Accumulate loss and count for averaging
        total_loss += class_loss
        count += 1

        if prev_class_logits is not None:
            total_loss = total_loss + loss_inter_class

        prev_class_logits =  logits_class_flat
        pre_class_mask = class_low_conf_mask_flat
        


    # Average loss over all classes
    if count > 0:
        total_loss /= count

    

    return total_loss



def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    if cfg['depth_type'] == 'v1':
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }

        encoder = 'vits' # or 'vitb', 'vits'
        model_depth = DepthAnything(model_configs[encoder])
        model_depth.load_state_dict(torch.load(f'/DepthV1/depth_anything_{encoder}14.pth'))
        model_depth = model_depth.cuda()
        local_rank = int(os.environ["LOCAL_RANK"])
        model_depth = torch.nn.parallel.DistributedDataParallel(
            model_depth, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
        )

        model_depth = model_depth.eval()
        for param in model_depth.parameters():
            param.requires_grad = False


    elif cfg['depth_type'] == 'v2':
        model_depth = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vits' # or 'vits', 'vitb', 'vitg'

        model_depth = DepthAnythingV2(**model_depth[encoder])
        model_depth.load_state_dict(torch.load(f'/Depth/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model_depth = model_depth.cuda()
        local_rank = int(os.environ["LOCAL_RANK"])

        model_depth = torch.nn.parallel.DistributedDataParallel(
            model_depth, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
        )

        model_depth = model_depth.eval()
        for param in model_depth.parameters():
            param.requires_grad = False



    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
        
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False


    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )

    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    cons_w_unsup = consistency_weight(final_w=cfg['consisteny_weight'], iters_per_epoch=len(trainloader_u),
                                        rampup_ends=int(cfg['epochs'])*10)
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_distill = AverageMeter()
        total_loss_consistency = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        start_time = time.time()
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()

            img_u_w_1 = img_u_w.clone()
            img_u_w_2 = img_u_w.clone()

            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w_first = pred_u_w.softmax(dim=1)
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            img_u_w_1[cutmix_box1.unsqueeze(1).expand(img_u_w_1.shape) == 1] = img_u_w.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_w.shape) == 1]
            img_u_w_2[cutmix_box2.unsqueeze(1).expand(img_u_w_2.shape) == 1] = img_u_w.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_w.shape) == 1]

            
            pred_x = model(img_x)
            pred_u_s, feature_u_s = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True)
            pred_u_s1, pred_u_s2 = pred_u_s.chunk(2)
            feature_s_1, feature_s_2 = feature_u_s.chunk(2)

            pred_depth_u_all,feature_depth_u_all = model_depth(torch.cat((img_u_w_1, img_u_w_2)))
            pred_depth_u_w_1,pred_depth_u_w_2 = pred_depth_u_all.chunk(2)  # 518 × 518


            # ##################局部深度信息指导特征信息##################
            mask_clone_1 = mask_u_w.clone()
            mask_clone_2 = mask_u_w.clone()
            ignore_mask_clone_1 = ignore_mask.clone()
            ignore_mask_clone_2 = ignore_mask.clone()
            conf_u_w_clone_1 = conf_u_w_first.clone()
            conf_u_w_clone_2 = conf_u_w_first.clone()
            mask_clone_1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            mask_clone_2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_clone_1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            ignore_mask_clone_2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            conf_u_w_clone_1[cutmix_box1.unsqueeze(1).expand(conf_u_w_first.shape) == 1] = conf_u_w_first.flip(0)[cutmix_box1.unsqueeze(1).expand(conf_u_w_first.shape) == 1]
            conf_u_w_clone_2[cutmix_box2.unsqueeze(1).expand(conf_u_w_first.shape) == 1] = conf_u_w_first.flip(0)[cutmix_box2.unsqueeze(1).expand(conf_u_w_first.shape) == 1]


            loss_depth1 = depth_difference_consistency_loss(feature_s_1,pred_depth_u_w_1,conf_u_w_clone_1,mask_clone_1,ignore_mask_clone_1,rank)
            loss_depth2 = depth_difference_consistency_loss(feature_s_2,pred_depth_u_w_2,conf_u_w_clone_2,mask_clone_2,ignore_mask_clone_2,rank)
            loss_depth = cfg['distill_loss_weight'] * (loss_depth1 + loss_depth2)
            # ##################局部深度信息指导特征信息##################

            # ##################全局深度信息指导特征信息##################
            # loss_depth1 = depth_difference_consistency_loss_global(feature_s_1,pred_depth_u_w_1)
            # loss_depth2 = depth_difference_consistency_loss_global(feature_s_2,pred_depth_u_w_2)
            # loss_depth = cfg['distill_loss_weight'] * (loss_depth1 + loss_depth2) / 2
            # ##################全局深度信息指导特征信息##################
            
            weight_u = cons_w_unsup(epoch=epoch,curr_iter=i)

            # # ##################深度图局部区域一致性到预测结果的一致性##################
            loss_uncertainty_consistency_1 = depth_guided_class_specific_consistency_loss(pred_u_s1, pred_depth_u_w_1,mask_clone_1, ignore_mask_clone_1)
            loss_uncertainty_consistency_2 = depth_guided_class_specific_consistency_loss(pred_u_s2, pred_depth_u_w_2,mask_clone_2, ignore_mask_clone_2)

            loss_uncertainty_consistency = 0.005 * (loss_uncertainty_consistency_1 + loss_uncertainty_consistency_2)
            # ######################################################################



            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]

            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            num_class = torch.unique(mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
            
            loss_u_s = (loss_u_s1 + loss_u_s2 + loss_depth + loss_uncertainty_consistency) / 2.0  
            
            loss = (loss_x + loss_u_s) / 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            total_loss_distill.update(loss_depth.item())
            if not isinstance(loss_uncertainty_consistency, float):
                total_loss_consistency.update(loss_uncertainty_consistency.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))


            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss distill: {:.12f}, Loss_consistency: {:.12f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_loss_distill.avg,total_loss_consistency.avg,total_mask_ratio.avg))
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"每轮程序运行时间: {execution_time / 60} min")
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)

        is_best = mIoU >= previous_best
        
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
