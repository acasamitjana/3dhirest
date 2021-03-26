#!/usr/bin/env bash
python train_inter_symmetric.py --modality MRI_IHC  --model bidir --mask
python train_inter_symmetric.py --modality MRI_NISSL  --model bidir --mask
python train_inter_symmetric.py --modality IHC_NISSL  --model bidir --mask
