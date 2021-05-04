#!/bin/bash --gpu=0
pip install -r requirements.txt

for MC in {"MC1","MC1","MC1"}
    do       			
	echo "Config: $MC"
	python main_obow.py --config="$MC"
	echo "CAMBIO PARAMETRI"

    done
echo "TERMINE DEL PROCESSO"
