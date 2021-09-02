#include <math.h>
#include <stdlib.h>
#include <stdio.h>

double tan1(double f) {
    return sin(f)/cos(f);
}

double *loop(double *arr, int n) {
    double *b = (double*)malloc(n * sizeof(double));
    for(int i = 0; i < n; i++) {
        b[i] = tan(arr[i]);
    }
    return b;
}

void freeArray(double *b) {
    free(b);
}