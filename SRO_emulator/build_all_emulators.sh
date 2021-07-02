#!/bin/bash

for var in Abar X3he X4li Xa Xd Xh Xn Xp Xt Zbar cs2 dedt dpderho dpdrhoe entropy gamma logenergy logpress mu_e mu_n mu_p muhat munu
do
sbatch build_emulator.sh $var
done
