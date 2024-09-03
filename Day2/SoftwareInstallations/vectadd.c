#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{

// Initialize arrays 
const long long size=1000000;
float a[size];
float b[size];
float c[size];
float d[size];

// Fill with a & b with random numbers between 0 and 100
int i;
for(i=0; i<size; i++)
{
  a[i]=rand();
  b[i]=rand();
  c[i]=rand();
}

for(i=0; i<size; i++)
{   
  d[i]=a[i]*b[i]+c[i];
}

printf("First element of vector c: %g\n", d[0]);

return 0;
}
