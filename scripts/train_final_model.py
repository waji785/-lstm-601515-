#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练最终生产模型（使用全部历史数据截至今日）
用法: python scripts/train_final_model.py [--stocks 2000] [--end-date 2026-08-04]
"""

import os
import sys
import argparse
import pandas as pd
import torch
from config.settings import *
from core.data_loader import load_all_stock_data
from core.trainer import train_and_save_model
from utils.common import set_seed
from utils.logger import setup_logger

logger = setup_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="训练最终生产模型")
    parser.add_argument('--stocks', type=int, default=6000,
                        help='用于训练的股票数量（默认6000）')
    parser.add_argument('--end-date', type=str, default=TODAY,
                        help=f'训练截止日期，格式 YYYY-MM-DD（默认 {TODAY}）')
    parser.add_argument('--model-path', type=str, default="model_final.pth",
                        help='模型保存路径（默认 model_final.pth）')
    parser.add_argument('--scaler-x', type=str, default="scaler_X_final.pkl",
                        help='X标准化器保存路径')
    parser.add_argument('--scaler-y', type=str, default="scaler_Y_final.pkl",
                        help='Y标准化器保存路径')
    parser.add_argument('--min-days', type=int, default=200,
                        help='股票最少数据天数（默认200）')
    parser.add_argument('--force', action='store_true',
                        help='如果模型已存在，强制覆盖（不提示）')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)

    # 检查是否已有模型文件
    if not args.force and os.path.exists(args.model_path):
        confirm = input(f"模型文件 {args.model_path} 已存在，是否覆盖？(y/n) ")
        if confirm.lower() != 'y':
            logger.info("操作取消")
            return

    logger.info("=" * 60)
    logger.info("🚀 开始训练最终生产模型（全部历史数据）")
    logger.info(f"  训练股票数: {args.stocks}")
    logger.info(f"  训练截止日期: {args.end_date}")
    logger.info(f"  最少数据天数: {args.min_days}")
    logger.info("=" * 60)

    # 加载全部数据（会自动过滤ST股票，因为 load_all_stock_data 默认 exclude_st=True）
    logger.info("正在加载股票数据...")
    df_all = load_all_stock_data(
        max_stocks=args.stocks,
        min_days=args.min_days,
        download_missing=True,
        start="2018-01-01",
        end=args.end_date,
        max_workers=1,      # 单线程避免连接问题
        force_full=False
    )
    if df_all is None or len(df_all) < 1000:
        logger.error("数据加载失败或数据量不足")
        return

    logger.info(f"数据加载完成，总样本数: {len(df_all)}")

    # 训练最终模型
    logger.info("开始训练...")
    model, scaler_X, scaler_y, _ = train_and_save_model(
        df=df_all,
        train_end_date=args.end_date,
        model_save_path=args.model_path,
        scaler_x_path=args.scaler_x,
        scaler_y_path=args.scaler_y
    )

    if model is not None:
        logger.info(f"✅ 最终模型已保存至: {args.model_path}")
        logger.info(f"   标准化器保存至: {args.scaler_x}, {args.scaler_y}")
        # 输出一些模型信息
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   模型参数总量: {total_params:,}")
    else:
        logger.error("训练失败")

if __name__ == "__main__":
    main()