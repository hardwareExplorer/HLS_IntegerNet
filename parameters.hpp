#ifndef PARAM
#define PARAM

#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

typedef ap_axiu <32,0,0,0> AXI_STREAM;

//Using union

typedef union f4{
	unsigned int 	uidata;
	float			 fdata;

}converter_f;




#define INT_SIZE 5 //number of bits allocated for integerized value




//-------------------LAYER_1------------------------------

//CONV_1

#define IN_H_1 22
#define IN_W_1 114
#define IN_D_1 1
#define K_H_1 22
#define K_W_1 5
#define K_D_1 1
#define K_CNT_1 16
#define STRIDE_V_1 1
#define STRIDE_H_1 2
#define OUT_D_1 K_CNT_1
#define OUT_H_1 ((IN_H_1-K_H_1)/STRIDE_V_1)+1
#define OUT_W_1 ((IN_W_1-K_W_1)/STRIDE_H_1)+1



//MAXPOOL_1

#define IN_H_P1 1
#define IN_W_P1 55
#define IN_D_P1 16
#define K_H_P1 1
#define K_W_P1 2
#define STRIDE_V_P1 1
#define STRIDE_H_P1 2
#define OUT_D_P1 IN_D_P1
#define OUT_H_P1 ((IN_H_P1-K_H_P1)/STRIDE_V_P1)+1
#define OUT_W_P1 ((IN_W_P1-K_W_P1)/STRIDE_H_P1)+1


//-------------------LAYER_2------------------------------

//CONV_2
#define IN_H_2 1
#define IN_W_2 27
#define IN_D_2 16
#define K_H_2 1
#define K_W_2 3
#define K_D_2 16
#define K_CNT_2 32
#define STRIDE_V_2 1
#define STRIDE_H_2 2
#define OUT_D_2 K_CNT_2
#define OUT_H_2 ((IN_H_2-K_H_2)/STRIDE_V_2)+1
#define OUT_W_2 ((IN_W_2-K_W_2)/STRIDE_H_2)+1

//MAXPOOL_2


#define IN_H_P2 1
#define IN_W_P2 13
#define IN_D_P2 32
#define K_H_P2 1
#define K_W_P2 2
#define STRIDE_V_P2 1
#define STRIDE_H_P2 2
#define OUT_D_P2 IN_D_P2
#define OUT_H_P2 ((IN_H_P2-K_H_P2)/STRIDE_V_P2)+1
#define OUT_W_P2 ((IN_W_P2-K_W_P2)/STRIDE_H_P2)+1



//-------------------LAYER_3------------------------------

//CONV_3
#define IN_H_3 1
#define IN_W_3 6
#define IN_D_3 32
#define K_H_3 1
#define K_W_3 3
#define K_D_3 32
#define K_CNT_3 64
#define STRIDE_V_3 1
#define STRIDE_H_3 1
#define OUT_D_3 K_CNT_3
#define OUT_H_3 ((IN_H_3-K_H_3)/STRIDE_V_3)+1
#define OUT_W_3 ((IN_W_3-K_W_3)/STRIDE_H_3)+1

//MAXPOOL_3


#define IN_H_P3 1
#define IN_W_P3 4
#define IN_D_P3 64
#define K_H_P3 1
#define K_W_P3 2
#define STRIDE_V_P3 1
#define STRIDE_H_P3 2
#define OUT_D_P3 IN_D_P3
#define OUT_H_P3 ((IN_H_P3-K_H_P3)/STRIDE_V_P3)+1
#define OUT_W_P3 ((IN_W_P3-K_W_P3)/STRIDE_H_P3)+1


//-------------------LAYER_4------------------------------

//DENSE_1

#define IN_H_D1 128
#define IN_W_D1 1
#define K_H_D1 128
#define K_W_D1 128
#define OUT_H_D1 K_H_D1
#define OUT_W_D1 IN_W_D1
//ACT0:none, 1:Sigmoid, 2:Softmax
#define ACT1 1


//-------------------LAYER_5------------------------------

//DENSE_2

#define IN_H_D2 128
#define IN_W_D2 1
#define K_H_D2 2
#define K_W_D2 128
#define OUT_H_D2 K_H_D2
#define OUT_W_D2 IN_W_D2
#define ACT2 2



#endif
