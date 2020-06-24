#!/bin/bash

python mnist.py --generate_adv --type plain
python mnist.py --generate_adv --type ACET
python mnist.py --generate_adv --type OE

python cifar10.py --generate_adv --type plain
python cifar10.py --generate_adv --type ACET
python cifar10.py --generate_adv --type OE

python svhn.py --generate_adv --type plain
python svhn.py --generate_adv --type ACET
python svhn.py --generate_adv --type OE

python cifar100.py --generate_adv --type plain
python cifar100.py --generate_adv --type ACET
python cifar100.py --generate_adv --type OE
