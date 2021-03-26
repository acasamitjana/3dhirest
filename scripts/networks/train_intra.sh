#!/usr/bin/env bash
python train_intra_symmetric.py --modality MRI  --model bidir --mask
python train_intra_symmetric.py --modality IHC  --model bidir --mask
python train_intra_symmetric.py --modality NISSL  --model bidir --mask
