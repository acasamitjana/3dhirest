#!/usr/bin/env bash
python predict_intra.py --modality MRI  --model bidir --epoch FI
python predict_intra.py --modality NISSL  --model bidir --epoch FI
python predict_intra.py --modality IHC  --model bidir --epoch FI
python predict_inter.py --modality MRI_IHC  --model bidir --epoch FI
python predict_inter.py --modality MRI_NISSL  --model bidir --epoch FI
python predict_inter.py --modality IHC_NISSL  --model bidir --epoch FI
