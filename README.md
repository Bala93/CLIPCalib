# Robust Calibration of Large Vision-Language Adapters (ECCV 2024)

## Installation
This repository requires to install the environment and datasets:
- follow [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) and PyTorch.
- run `pip install -r requirements.txt` under `CLIPCALIB/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated).
- follow [DATASETS.md](DATASETS.md) to install the datasets.

*PS: You can also follow [CoOp](https://github.com/KaiyangZhou/CoOp) to perform the installation.*

## Usage
We present the basic usage here.

(a) TR -- ZS-Norm:
- `bash scripts/adapt_zs_norm.sh 0 imagenet SGD_lr1e-1_B256_ep300 16 TR none RN50`
- `bash scripts/eval_zs_norm.sh 0 imagenetv2 SGD_lr1e-1_B256_ep300 16 TR none RN50`

(b) TR -- Penalty:
- `bash scripts/adapt_zs_pen.sh.sh 0 imagenet SGD_lr1e-1_B256_ep300 16 TR none RN50`
- `bash scripts/eval_zs_pen.sh.sh 0 imagenetv2 SGD_lr1e-1_B256_ep300 16 TR none RN50`

(c) TR -- SaLS:
- `bash scripts/adapt.sh 0 imagenet SGD_lr1e-1_B256_ep300 16 TR none RN50`
- `bash scripts/eval.sh 0 imagenetv2 SGD_lr1e-1_B256_ep300 16 TR none RN50`
- `bash scripts/eval_zs.sh 0 imagenetv2 SGD_lr1e-1_B256_ep300 16 ZS none RN50`
- The logits of the predictions are renormalized using the following snippet. 

  ```python
  logits_tr = (logits_tr - min_logits_tr)/ (max_logits_tr - min_logits_tr)
  logits_tr = logits_tr * (max_logits_zs - min_logits_zs) + min_logits_zs
  ```

## Notes
The integration of proposed calibration techniques is also available for prompt learning and test time prompt tuning in the branches.

## Acknowledgment
This repository is mainly based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [TaskRes](https://github.com/geekyutao/TaskRes) code base. We sincerely thank prior authors on this topic for his awesome code base.
