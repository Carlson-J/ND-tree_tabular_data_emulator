#!/bin/bash

for var in cs2 dedt dpderho dpdrhoe entropy logpress mu_e mu_n mu_p muhat munu
do
sbatch build_emulator.sh $var
done

#Abar Xa Xh Xn Xp Zbar logenergy