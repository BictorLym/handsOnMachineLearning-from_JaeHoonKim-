#include <iostream> 
using namespace std; 
 
int main() {     
    volatile int num1 = 8;     
    volatile int num2 = 8;     
    volatile int num3, num4, num5; 
 
    num1++;     
    num3 = --num1;     
    --num2;     
    num4 = num2++; 
 
    num5 = num1-- + ++num4;     
    cout << num1 << '\t' << num2 << '\t' << num3 << '\t' << num4 << '\t' << num5 << endl;    
    return 0; 
} 