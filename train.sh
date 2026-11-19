#!/usr/bin/env bash

set -euo pipefail

## Goodwin oscillator
#python src/train.py --config src/configs/goodwin/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/goodwin/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/goodwin/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/goodwin/koopman_baseline_grassmann_true

## Lorenz-63 system
#python src/train.py --config src/configs/lorenz/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/lorenz/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/lorenz/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/lorenz/koopman_baseline_grassmann_true

## Dysts chaotic systems
#python src/train.py --config src/configs/chua/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/chua/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/chua/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/chua/koopman_baseline_grassmann_true
#python src/train.py --config src/configs/dadras/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/dadras/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/dadras/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/dadras/koopman_baseline_grassmann_true
#python src/train.py --config src/configs/lu_chen_cheng/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/lu_chen_cheng/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/lu_chen_cheng/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/lu_chen_cheng/koopman_baseline_grassmann_true
#python src/train.py --config src/configs/shimizu_morioka/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/shimizu_morioka/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/shimizu_morioka/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/shimizu_morioka/koopman_baseline_grassmann_true

## Duffing double-well oscillator
#python src/train.py --config src/configs/duffing/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/duffing/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/duffing/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/duffing/koopman_baseline_grassmann_true

## Competitive Lotka–Volterra
#python src/train.py --config src/configs/lotka_volterra/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/lotka_volterra/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/lotka_volterra/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/lotka_volterra/koopman_baseline_grassmann_true

## Reaching
#python src/train.py --config src/configs/monkey_reaching/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/monkey_reaching/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/monkey_reaching/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/monkey_reaching/koopman_baseline_grassmann_true

## PAMY2 pneumatic robot arm
#python src/train.py --config src/configs/PAMy2/koopman_baseline.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --no-grassmann --outdir outputs/PAMy2/koopman_baseline_grassmann_false
#python src/train.py --config src/configs/PAMy2/koopman_baseline_grassmann.yaml --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder --grassmann --outdir outputs/PAMy2/koopman_baseline_grassmann_true
