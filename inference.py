from __future__ import print_function, division
import argparse
from loguru import logger as loguru_logger
import random
from core.Networks import build_network
import sys
sys.path.append('core')
from PIL import Image
import os
import numpy as np
import torch
from utils import flow_viz
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
from inference import inference_core_skflow as inference_core
import cv2  # 新增 OpenCV 读取视频
import gc    # 用于手动清理资源
from torchvision import transforms


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import cv2  # 新增 OpenCV 读取视频
import gc    # 用于手动清理资源
from torchvision import transforms

@torch.no_grad()
def inference(cfg):
    model = build_network(cfg).cuda()
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in list(ckpt_model.keys()):
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=True)
        else:
            model.load_state_dict(ckpt_model, strict=True)

    model.eval()
    processor = inference_core.InferenceCore(model, config=cfg)

    # 打开视频流（支持摄像头或视频文件）
    cap = cv2.VideoCapture(cfg.seq_dir)  # 若为摄像头输入可用 0

    prev_frame_tensor = None
    flow_prev = None
    frame_idx = 0

    if not os.path.exists(cfg.vis_dir):
        os.makedirs(cfg.vis_dir)

    transform = transforms.ToTensor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理：BGR -> RGB, 转成 tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb).unsqueeze(0).cuda()  # 1, 3, H, W

        if prev_frame_tensor is None:
            prev_frame_tensor = frame_tensor
            continue

        # 拼接两个连续帧
        input_pair = torch.stack([prev_frame_tensor[0], frame_tensor[0]], dim=0).unsqueeze(0)  # 1, 2, 3, H, W
        padder = InputPadder(input_pair.shape)
        input_pair = padder.pad(input_pair)
        input_pair = 2 * (input_pair / 255.0) - 1.0

        flow_low, flow_pre = processor.step(
            input_pair,
            end=False,
            add_pe=('rope' in cfg and cfg.rope),
            flow_init=flow_prev
        )

        flow_pre = padder.unpad(flow_pre[0]).detach().cpu()
        if 'warm_start' in cfg and cfg.warm_start:
            flow_prev = forward_interpolate(flow_low[0])[None].cuda()

        # 可视化与保存
        flow_img = flow_viz.flow_to_image(flow_pre.permute(1, 2, 0).numpy())
        cv2.imwrite(f"{cfg.vis_dir}/flow_{frame_idx:04d}.png", cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))

        # 释放内存（防止内存泄露）
        del input_pair, flow_low, flow_pre
        torch.cuda.empty_cache()
        gc.collect()

        prev_frame_tensor = frame_tensor
        frame_idx += 1

    cap.release()
    print("Inference finished.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MemFlowNet', choices=['MemFlowNet', 'MemFlowNet_T'], help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--seq_dir', default='default')
    parser.add_argument('--vis_dir', default='default')

    args = parser.parse_args()
    if args.name == "MemFlowNet":
        if args.stage == 'things':
            from configs.things_memflownet import get_cfg
        elif args.stage == 'sintel':
            from configs.sintel_memflownet import get_cfg
        elif args.stage == 'spring_only':
            from configs.spring_memflownet import get_cfg
        elif args.stage == 'kitti':
            from configs.kitti_memflownet import get_cfg
        else:
            raise NotImplementedError
    elif args.name == "MemFlowNet_T":
        if args.stage == 'things':
            from configs.things_memflownet_t import get_cfg
        elif args.stage == 'things_kitti':
            from configs.things_memflownet_t_kitti import get_cfg
        elif args.stage == 'sintel':
            from configs.sintel_memflownet_t import get_cfg
        elif args.stage == 'kitti':
            from configs.kitti_memflownet_t import get_cfg
        else:
            raise NotImplementedError

    cfg = get_cfg()
    cfg.update(vars(args))

    # initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    inference(cfg)
