EPOCHS = 120
MOSAIC = 0.8
MIXUP = 0.1
OPTIMIZER = "AdamW"
MOMENTUM = 0.9
LR0 = 0.001
LRF = 0.0001
SINGLE_CLS = False
BATCH = 16
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
WEIGHT_DECAY = 0.001
WARMUP_EPOCHS = 5.0

import argparse
from ultralytics import YOLO
import os
import sys
from ultralytics.utils.torch_utils import select_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    # mosaic
    parser.add_argument(
        "--mosaic", type=float, default=MOSAIC, help="Mosaic augmentation"
    )
    # mixup
    parser.add_argument(
        "--mixup", type=float, default=MIXUP, help="Mixup augmentation"
    )
    # optimizer
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="Optimizer")
    # momentum
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum")
    # lr0
    parser.add_argument("--lr0", type=float, default=LR0, help="Initial learning rate")
    # lrf
    parser.add_argument("--lrf", type=float, default=LRF, help="Final learning rate")
    # single_cls
    parser.add_argument(
        "--single_cls", type=bool, default=SINGLE_CLS, help="Single class training"
    )
    # batch
    parser.add_argument(
        "--batch", type=int, default=BATCH, help="Batch size"
    )
    # hsv_h
    parser.add_argument(
        "--hsv_h", type=float, default=HSV_H, help="Hue augmentation"
    )
    # hsv_s
    parser.add_argument(
        "--hsv_s", type=float, default=HSV_S, help="Saturation augmentation"
    )
    # hsv_v
    parser.add_argument(
        "--hsv_v", type=float, default=HSV_V, help="Value augmentation"
    )
    # weight_decay
    parser.add_argument(
        "--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay"
    )
    # warmup_epochs
    parser.add_argument(
        "--warmup_epochs", type=float, default=WARMUP_EPOCHS, help="Warmup epochs"
    )
    print("used device is : ")
    device = select_device("cuda")
    print(device)
    print("----------")

    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    model = YOLO(os.path.join(this_dir, "yolov3u.pt"))
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        device=0,
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        mixup=args.mixup,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        batch=args.batch,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=True,
    )