#!/bin/bash

arch=$1

echo "ARCHITECTURE: $arch"
echo ""

echo "correlation: none"
python test.py --model_name pythia70m --sae sae_results/gemma-2-2b/${arch}_uniform/latest_ckpt.pt
echo "==============="
echo ""

for corr in {0.1,0.2,0.5,0.9,1.0}; do
    echo "correlation: $corr"
    python test.py --model_name pythia70m --sae sae_results/gemma-2-2b/${arch}_corr_${corr}/latest_ckpt.pt
    echo "=============="
    echo ""
done