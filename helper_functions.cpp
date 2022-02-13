#ifndef HELPER
#define HELPER

#include "definitions.hpp"

float abs_f(float x)
{
	return (x>0) ? x : -x;
}


//Sigmoid
//ref: https://hackaday.io/page/5331-sigmoid-function

float sigmoid(float x) {
     float result;
     result = 1 / (1 + exp(-x));
     return result;
}




#endif
