import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
import numpy as np

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            if hasattr(v, "shape"):
                log(f'  {k}: shape={tuple(v.shape)}')
            else:
                log(f'  {k}: (non-tensor: {type(v).__name__})')


    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler,drop_last=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'kvasir':
        metric_fn = utils.calc_kvasir
        metric1, metric2, metric3, metric4 = 'dice', 'iou', 'none', 'none'
        

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    
    val_metric1 = 0
    val_metric2 = 0
    val_metric3 = 0
    val_metric4 = 0
    cnt = 0
    
    for batch in loader:
        for k, v in batch.items():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch[k] = move_to_device(v, device)

        inp = batch['inp']

        # pred = torch.sigmoid(model.infer(inp))
        if hasattr(model, 'module'):
            pred = torch.sigmoid(model.module.infer(inp))
        else:
            pred = torch.sigmoid(model.infer(inp))
        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]
        
        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1 += (result1 * pred.shape[0])
        val_metric2 += (result2 * pred.shape[0])
        val_metric3 += (result3 * pred.shape[0])
        val_metric4 += (result4 * pred.shape[0])     
        cnt += pred.shape[0]
        if pbar is not None:
            pbar.update(1)
    val_metric1 = torch.tensor(val_metric1).cuda()
    val_metric2 = torch.tensor(val_metric2).cuda()
    val_metric3 = torch.tensor(val_metric3).cuda()
    val_metric4 = torch.tensor(val_metric4).cuda()
    cnt = torch.tensor(cnt).cuda()
    dist.all_reduce(val_metric1)
    dist.all_reduce(val_metric2)
    dist.all_reduce(val_metric3)
    dist.all_reduce(val_metric4)
    dist.all_reduce(cnt)
          
    if pbar is not None:
        pbar.close()
    
    return val_metric1.item()/cnt, val_metric2.item()/cnt, val_metric3.item()/cnt, val_metric4.item()/cnt, metric1, metric2, metric3, metric4


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    edge_loss_list = []
    for batch in train_loader:
        inp = batch['inp']
        gt = batch['gt']
        edge_gt = batch.get('edge', None)  # 预计算的边缘 GT（可选）

        model.module.optimizer.zero_grad()

        loss, edge_loss = model(inp, gt, edge_gt=edge_gt)

        loss.backward()

        model.module.optimizer.step()

        batch_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, loss)
        loss_list.extend(batch_loss)

        batch_edge_loss = [torch.zeros_like(edge_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_edge_loss, edge_loss)
        edge_loss_list.extend(batch_edge_loss)
        
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    edge_loss = [i.item() for i in edge_loss_list]
    return mean(loss), mean(edge_loss)

def move_to_device(x, device):
    """把 x 移到 device 上（支持 Tensor, numpy, list/tuple/dict 里含 Tensor/np）。
    对无法转换的类型（str, metadata）保持原样返回。
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    if isinstance(x, (list, tuple)):
        # 如果列表中全是 tensor/np，可以 stack；否则尝试逐元素 move 并返回同结构的 list
        moved = []
        all_tensors = True
        for e in x:
            if isinstance(e, torch.Tensor) or isinstance(e, np.ndarray):
                moved.append(move_to_device(e, device))
            elif isinstance(e, (int, float)):
                # convert scalars to tensor
                moved.append(torch.tensor(e).to(device))
            else:
                all_tensors = False
                break
        if all_tensors:
            # 返回 list（保持原类型），不要强制 stack ——上层可能期待 list
            return type(x)(moved)
        else:
            # 对不可移动的元素，返回原始列表（不移动这些元素）
            # 如果你想要尽可能移动成功的元素，可以只替换可移动的部分；这里选择保守行为
            return x
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    # 默认：不能移动（比如 str, tuple metadata），直接返回原值
    return x

def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    
    model.optimizer = optimizer
    
    lr_scheduler = CosineAnnealingLR(optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.cuda()

    ckpt = torch.load(config['sam_checkpoint'], map_location="cpu")
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
        
    new_state_dict = {}

    ref_state_dict = model.state_dict()

    if args.local_rank == 0:
        print(f"Loading custom checkpoint with 'detector.backbone' prefix...")

    for k, v in ckpt.items():

        if k.startswith("detector.backbone."):
            new_k = k.replace("detector.backbone.", "image_encoder.")

        elif "mask_decoder" in k:
            suffix = k.split("mask_decoder.")[-1]
            new_k = f"mask_decoder.{suffix}"
            
        elif "pe_layer" in k:
            suffix = k.split("pe_layer.")[-1]
            new_k = f"pe_layer.{suffix}"

        elif "no_mask_embed" in k:
            new_k = "no_mask_embed.weight"

        else:
            new_k = k

        if new_k in ref_state_dict:
            ref_shape = ref_state_dict[new_k].shape
            if v.shape != ref_shape:
                if args.local_rank == 0:
                    print(f"Warning: Skipping {new_k} due to shape mismatch. "
                          f"Ckpt: {v.shape} vs Model: {ref_shape}")
                continue

        if new_k:
            new_state_dict[new_k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)

    if args.local_rank == 0:
        print(f"\nLoad result: {len(msg.missing_keys)} missing keys.")
        if len(msg.missing_keys) > 0:
             print("Sample missing keys:", msg.missing_keys[:3])

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    if args.local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )

        
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        
        train_loss_G, train_edge_loss = train(train_loader, model)
        lr_scheduler.step()

        if args.local_rank == 0:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalars('loss', {'train G': train_loss_G, 'edge': train_edge_loss}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.module.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            save(config, model.module, save_path, 'last')
            
            # 保存第5个和第10个epoch的结果
            if epoch == 5 or epoch == 10:
                save(config, model.module, save_path, f'epoch_{epoch}')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(
                val_loader, model, eval_type=config.get('eval_type')
            )

            if args.local_rank == 0:
                log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
                log_info.append('train G: loss={:.4f}'.format(train_loss_G))
                log_info.append('train edge: loss={:.4f}'.format(train_edge_loss))
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model.module, save_path, 'best')
                else:
                    if result2 < max_val_v:
                        max_val_v = result2
                        save(config, model.module, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()
            dist.barrier()

def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/home/ubuntu/public_c/crl/sam3_adapter/SAM-Adapter-PyTorch/configs/cod-sam-vit-l.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    # parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--local-rank", type=int, default=0, help="")

    args = parser.parse_args()
    

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
