#!/bin/bash --gpu=0
# pip install -r requirements.txt

for MC in {"MC25","prova_test"}
    do       			
	echo "Config: $MC"
	python train.py --config="$MC"
	echo "CAMBIO PARAMETRI"

    done
echo "TERMINE DEL PROCESSO"
