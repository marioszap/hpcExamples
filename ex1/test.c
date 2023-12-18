#include <stdio.h>
#include "omp.h"



static long num_steps = 1e4;
double step;

int main() {
    int i;
    double x, pi, sum = 0.;
    step = 1./(double) num_steps;

    for(i=0; i<num_steps; i++){
        x = (i+0.5) * step;
        sum = sum + 4./(1. +x*x);
    }
    pi = step*sum;

    printf("%f \n", pi);

}
