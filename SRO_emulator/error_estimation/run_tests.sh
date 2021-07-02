#!/bin/bash

for var in Abar Xa Xh Xl Xn Xp Zbar Zlbar cs2 dedt dpderho dpdrhoe entropy gamma logenergy logpress meffn meffp mu_e mu_n mu_p muhat munu r u 
do
sbatch compute_error.sh $var
done
