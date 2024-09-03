#include <stdio.h>

double square(double a) {
    return a*a;
}

int main() {
    double a, square_a;
    printf("Enter one number: ");
    scanf("%lf", &a);  
 
    // Calculating square
    square_a = square(a);

    // %.2lf displays number up to 2 decimal point
    printf("Square = %.2lf\n", square_a);
    
    return 0;
}
