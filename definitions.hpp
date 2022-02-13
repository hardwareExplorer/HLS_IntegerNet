#ifndef INT_OP
#define INT_OP

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "hls_math.h"
#include <stdio.h>
#include "ap_int.h"
#include "parameters.hpp"
#include "datafilepaths.hpp"


#include <math.h>
#include <assert.h>

#include <iostream>

using namespace std;



//Absolute function
#ifdef ABS
#undef ABS
#endif
#define ABS(n) ((n < 0) ? -n : n)


float abs_f(float x);
float sigmoid(float x);


//flatten
template<typename T, int mtx_d, int mtx_h, int mtx_w>
void flatten(T (&in1)[mtx_d][mtx_h][mtx_w], T (&out1)[mtx_d*mtx_h*mtx_w][1])
{
	int i,j,k;

	for(k=0;k<mtx_d;k++)
		for(i=0;i<mtx_h;i++)
			for(j=0;j<mtx_w;j++)
			{
				out1[k*(mtx_h*mtx_w) + i*mtx_w + j][0] = in1[k][i][j];
			}
}

////////////////////////////////////////////////////////////////////////
/////// Conv & Maxpool////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
template<int in_h,int  in_w, int  in_d, int  k_h, int  k_w, int  k_d, int  k_cnt, int out_h, int out_w >
void conv_int(float (&in_map)[in_d][in_h][in_w], float scalar_wt[k_cnt], ap_int<INT_SIZE+2> (&wt)[k_cnt][k_d][k_h][k_w], int  st_v, int  st_h, float (&out_map)[k_cnt][out_h][out_w] )

{
	int m,n,k,i,j,p,q,r;
	float scalar_in;
	float win[k_d][k_h][k_w];
	ap_int<INT_SIZE+2> in_int[in_d][in_h][in_w];

	int acc;
	float max;
	float scalar_acc;


	//output feature map is produced in its channel wise (an extracted window is convolved against all the kernels in that layer thus producing output in 3rd dim)
			//1.Extract window

			integerize(in_map,in_int);
			//print_mtx(in_int,1);

			for(m=0,q=0; m<(in_h-k_h)+1; m=m+st_v,q++)//vertical count
				for(n=0,r=0; n<(in_w-k_w)+1; n=n+st_h,r++)//horizontal count
				{


					//1. Extract scalar
					scalar_acc = 0;
						for (i=0; i <k_h; i++)
							for(j=0; j<k_w; j++)
							{
								max = fabs(in_map[0][m+i][n+j]);
								for(k=0; k<k_d; k++)
								{
									if(fabs(in_map[k][m+i][n+j]) > max)
										max = fabs(in_map[k][m+i][n+j]);
								}

								scalar_acc += max;
							}

						scalar_acc /= 32;
						cout<<"\t "<<scalar_acc;


					//2. Convovle the inegerized matrix against all the kernels: produces channel wise output in OFM

					for(p=0;p<k_cnt;p++)
					{
						acc=0;

						for(k=0; k<k_d; k++)
							for (i=0; i <k_h; i++)
								for(j=0; j<k_w; j++)
								{
									acc += in_int[k][m+i][n+j] * wt[p][k][i][j];
									}

								//4. Multiply with the in and wt scalar

								out_map[p][q][r] = scalar_acc * scalar_wt[p] * acc;


						//5. Activation: reLu
						if(out_map[p][q][r]<0)
							out_map[p][q][r] = 0;

				}


				}//horizontal count repeat


}

//-----------------------------------conv maxpool : REAL ---------------------------------------------------------------

template<int in_h,int  in_w, int  in_d, int  k_h, int  k_w, int  k_d, int  k_cnt, int out_h, int out_w >
void conv_real(float (&in_map)[in_d][in_h][in_w], float (&wt)[k_cnt][k_d][k_h][k_w], int  st_v, int  st_h, float (&out_map)[k_cnt][out_h][out_w] )

{
	int m,n,k,i,j,p,q,r;
	//output feature map is produced in its channel wise (an extracted window is convolved against all the kernels in that layer thus producing output in 3rd dim)
			//1.Extract window
			for(m=0,q=0; m<(in_h-k_h)+1; m=m+st_v,q++)//vertical count
				for(n=0,r=0; n<(in_w-k_w)+1; n=n+st_h,r++)//horizontal count
				{
				for(p=0;p<k_cnt;p++)
				{
					out_map[p][q][r]=0;

					for(k=0; k<k_d; k++)
						for (i=0; i <k_h; i++)
							for(j=0; j<k_w; j++)
							{
								out_map[p][q][r]+= in_map[k][m+i][n+j]*wt[p][k][i][j];
									//cout<<"\t"<<in_map[k][m+i][n+j]<<"\t*\t"<<wt[p][k][i][j]<<"\t"<<in_map[k][m+i][n+j]*wt[p][k][i][j]<<"\t"<<out_map[p][q][r] ;

							}
							//cout<<"\n";

					//Activation: reLu
					if(out_map[p][q][r]<0)
						out_map[p][q][r] = 0;

				}


				}//horizontal count repeat


}

template<int in_h,int  in_w, int  in_d>
void BN_real(float (&in_map)[in_d][in_h][in_w], float mean[in_d], float variance[in_d], float beta[in_d], float gamma[in_d] ){

	for(int k=0; k<in_d; k++)
		for (int i=0; i <in_h; i++)
		{
			for(int j=0; j<in_w; j++)
			{

				//BN the input
				in_map[k][i][j] = (in_map[k][i][j] - mean[k])/sqrt(variance[k]);
				//scale and shift
				in_map[k][i][j] = in_map[k][i][j]* gamma[k] + beta[k];

			}
		}



}



template<int in_h,int  in_w, int  in_d, int out_h, int out_w >
void maxpool_real(float (&in_map)[in_d][in_h][in_w], int k_h, int k_w, int  st_v, int  st_h, float (&out_map)[in_d][out_h][out_w] )

{
	int m,n,k,i,j,q,r;
	float max;

	//output feature map is produced in its channel wise (an extracted window is convolved against all the kernels in that layer thus producing output in 3rd dim)

			for(m=0,q=0; m<(in_h-k_h)+1; m=m+st_v,q++)//vertical count
				for(n=0,r=0; n<(in_w-k_w)+1; n=n+st_h,r++)//horizontal count
				{

					for(k=0; k<in_d; k++)
					{
						max = in_map[k][m][n];
						for (i=0; i <k_h; i++)
						{
							for(j=0; j<k_w; j++)
							{
								if(in_map[k][m+i][n+j] > max)
									max = in_map[k][m+i][n+j];
								//cout<<"\t"<<max<<"\t" ;

							}

						}

						out_map[k][q][r] = max;
						//cout<<"\t"<<out_map[k][q][r]<<"\t" ;
					}

					//cout<<"\n";

				}

}








////////////////////////////////////////////////////////////////////////
/////// Mtx mult top////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
template<int mtx_m>
void softmax(float (&in)[mtx_m][1], float (sum))
{
	for (int i=0; i<mtx_m; i++)
	{
		cout <<"i: "<< i<< " sum  "<<sum<<"  in : "<<in[i][0];
		in[i][0] = exp(in[i][0])/sum;
		cout<<"Softmax of "<<i<<"  "<< in[i][0];
	}

}





template<int mtx_m, int mtx_n>
//(input_mtx nxp, weight scalar, weight_int mxn, out_mtx nxp)
void dense1d_top_int(float (&MxA)[mtx_n], float scalarB,ap_int<INT_SIZE+2> (&MxB_int)[mtx_m][mtx_n],  float (&MxC)[mtx_m], int ACT){

	float scalarA;
	ap_int<INT_SIZE+2> MxA_int[mtx_n];

	scalarA = integerize_1d(MxA,MxA_int);

	int_int_mx_mult1d(scalarA, MxA_int, scalarB,MxB_int,  MxC, ACT);
}


// for real just call is enough



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Matrix Multiplication



template<int mtx_m, int mtx_n>
//A input nxp, B weight mxn, C = BxA mxp
void int_int_mx_mult1d(float scalarA, ap_int<INT_SIZE+2> (&MxA_int)[mtx_n], float scalarB, ap_int<INT_SIZE+2> (&MxB_int)[mtx_m][mtx_n], float (&MxC)[mtx_m], int ACT )
{


	int i,j,k;
	ap_int<INT_SIZE+2+9>  acc = 0; //saves 17bits from standard int
	float scalarC =  scalarA*scalarB;

	//cout<<"Acc out";
	for(i=0; i<mtx_m; i++)
	{

				for(k=0;k<mtx_n;k++)
				{
					acc = acc +  MxB_int[i][k]*MxA_int[k];
					//cout<<"\n"<<acc;
				}


				MxC[i] = scalarC * acc;
				//Activation:
				//If ACT = 0 -> Sigmoid, ACT=1 ->Softmax
				if(ACT==1)
					MxC[i] = sigmoid(MxC[i]);
				acc =0;

	}

	//if(ACT==2)
	//	softmax(MxC);



}

template<int mtx_m, int mtx_n>
//A input nxp, B weight mxn, C = BxA mxp
void dense1d_top_real(float (&MxA)[mtx_n], float (&MxB)[mtx_m][mtx_n], float (&MxC)[mtx_m], int ACT )
{

	int i,j,k;

		for(i=0; i<mtx_m; i++)
		{
					MxC[i] = 0;
					for(k=0;k<mtx_n;k++)
					{
						MxC[i] = MxC[i] +  MxB[i][k]*MxA[k];

						if(ACT==1)
							MxC[i] = sigmoid(MxC[i]);

					}

		}


	//	if(ACT==2)
		//	softmax(MxC);


}

//-----2D---------

template<int mtx_m, int mtx_n, int mtx_p>
//(input_mtx nxp, weight scalar, weight_int mxn, out_mtx nxp)
void dense_top_int(float (&MxA)[mtx_n][mtx_p], float scalarB,ap_int<INT_SIZE+2> (&MxB_int)[mtx_m][mtx_n],  float (&MxC)[mtx_m][mtx_p], int ACT){

	float scalarA;
	ap_int<INT_SIZE+2> MxA_int[mtx_n][mtx_p];

	scalarA = integerize_2d(MxA,MxA_int);

	int_int_mx_mult(scalarA, MxA_int, scalarB,MxB_int,  MxC, ACT);
}


// for real just call is enough



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Matrix Multiplication

template<int mtx_m, int mtx_n,int mtx_p>
//A input nxp, B weight mxn, C = BxA mxp
void int_int_mx_mult(float scalarA, ap_int<INT_SIZE+2> (&MxA_int)[mtx_n][mtx_p], float scalarB, ap_int<INT_SIZE+2> (&MxB_int)[mtx_m][mtx_n], float (&MxC)[mtx_m][mtx_p], int ACT )
{


	int i,j,k;
	ap_int<INT_SIZE+2+9>  acc = 0; //saves 17bits from standard int
	float scalarC =  scalarA*scalarB;
	float sum=0;

	//cout<<"Acc out";
	for(i=0; i<mtx_m; i++)
	{
		for(j=0; j<mtx_p; j++)
			{
				for(k=0;k<mtx_n;k++)
				{
					acc = acc +  MxB_int[i][k]*MxA_int[k][j];
					//cout<<"\n"<<acc;
				}


				MxC[i][j] = scalarC * acc;
				//Activation:
				//If ACT = 0 -> Sigmoid, ACT=1 ->Softmax
				if(ACT==1)
					MxC[i][j] = sigmoid(MxC[i][j]);
				else if (ACT ==2)
					sum += exp( MxC[i][j]);
				acc =0;
			}
	}

	if(ACT==2)
		softmax(MxC,sum);




}

template<int mtx_m, int mtx_n,int mtx_p>
//A input nxp, B weight mxn, C = BxA mxp
void dense_top_real(float (&MxA)[mtx_n][mtx_p], float (&MxB)[mtx_m][mtx_n], float (&MxC)[mtx_m][mtx_p], int ACT )
{

	int i,j,k;
	float sum=0;

		for(i=0; i<mtx_m; i++)
		{
				for(j=0; j<mtx_p; j++)
				{
					MxC[i][j] = 0;
					for(k=0;k<mtx_n;k++)
					{
						MxC[i][j] = MxC[i][j] +  MxB[i][k]*MxA[k][j];

					}

					if(ACT==1)
						MxC[i][j] = sigmoid(MxC[i][j]);
					else if (ACT ==2){
						sum += exp( MxC[i][j]);
						//cout<<"\nMxC "<<MxC[i][j]<<"  sum : "<< sum;
					}

				}

		}


	if(ACT==2)
		softmax(MxC, sum);


}



template<int mtx_m, int mtx_n,int mtx_p>
//A input nxp, B weight mxn, C = BxA mxp
void dense_top_pureint(ap_int<INT_SIZE+2> (&MxA)[mtx_n][mtx_p], ap_int<INT_SIZE+2> (&MxB)[mtx_m][mtx_n], ap_int<INT_SIZE+3> MxC[mtx_m][mtx_p] )
{


	int i,j,k;
	int acc = 0;

	for(i=0; i<mtx_m; i++)
			for(j=0; j<mtx_p; j++)
			{
				for(k=0;k<mtx_n;k++)
				{
					acc = acc +  MxB[i][k]*MxA[k][j];

				}
				MxC[i][j] = acc;
				acc =0;
			}

}


////////////////////////////////////////////////////////////////////////
///////////////// MAC tops /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
template<int mtx_m, int mtx_n,int mtx_k, int k_cnt>
void mac_top_int (float (&MxA)[mtx_k][mtx_m][mtx_n], ap_int<INT_SIZE+2> (&MxB_int)[k_cnt][mtx_k][mtx_m][mtx_n], float scalarB[k_cnt], float  mac_int[k_cnt])
{


		int k;
		float scalarA;
		ap_int<INT_SIZE+2> MxA_int[mtx_k][mtx_m][mtx_n];

		scalarA = integerize(MxA,MxA_int);

		for(k=0;k<k_cnt;k++)
		{
	    mac_int[k] =  int_int_mx_mac(scalarA,MxA_int,scalarB[k],MxB_int[k]);
		}


}

template<int mtx_m, int mtx_n,int mtx_k, int k_cnt>
void mac_top_real (float (&MxA)[mtx_k][mtx_m][mtx_n], float (&MxB)[k_cnt][mtx_k][mtx_m][mtx_n], float mac_real[k_cnt])
{

	int k;

	for(k=0;k<k_cnt;k++)
	{
		mac_real[k] = real_mx_mac (MxA,MxB[k]);
	}

}






/// TO_DO : S
//How fp multiplication is avoided?!!
//Try other rounding techniques
//Try other quantization techniques
//Try other correction techniques: improvising scalar factor

///--------------- MAC OPERATION ---------------
//(mxn) * (nxk) -> (mxk)


//---Single/multiple kernel Integerized MAC with A real

template<int mtx_h, int mtx_w, int mtx_d>
float int_mx_mac(float (&MxA)[mtx_d][mtx_h][mtx_w], float scalarB, ap_int<INT_SIZE+2> (&MxB_int)[mtx_d][mtx_h][mtx_w])
{

	float scalarA;
	ap_int<INT_SIZE+2> MxA_int [mtx_d][mtx_h][mtx_w];

	int i,j,k;
	int mac = 0;

	scalarA = integerize (MxA,MxA_int);

	for(k=0; k<mtx_d; k++)
			for(i=0; i<mtx_h; i++)
				for(j=0;j<mtx_w;j++)
				{
					mac = mac +  MxA_int[k][i][j]*MxB_int[k][i][j];
				}

	return scalarA * scalarB * mac;
}

//Single/Multiple kernel integerized MAC both A & B are integerized
template<int mtx_h, int mtx_w, int mtx_d>
float int_int_mx_mac(float scalarA, ap_int<INT_SIZE+2> (&MxA_int)[mtx_d][mtx_h][mtx_w], float scalarB, ap_int<INT_SIZE+2> (&MxB_int)[mtx_d][mtx_h][mtx_w])
{


	int i,j,k;
	int mac = 0;


	for(k=0; k<mtx_d; k++)
			for(i=0; i<mtx_h; i++)
				for(j=0;j<mtx_w;j++)
				{
					mac = mac +  MxA_int[k][i][j]*MxB_int[k][i][j];
				}

	return scalarA * scalarB * mac;
}

//---Real MAC

template<int mtx_h, int mtx_w, int mtx_d>
float real_mx_mac(float (&MxA)[mtx_d][mtx_h][mtx_w], float (&MxB)[mtx_d][mtx_h][mtx_w])
{


	int i,j,k;
	float mac = 0;

	for(k=0; k<mtx_d; k++)
			for(i=0; i<mtx_h; i++)
			{
				for(j=0;j<mtx_w;j++)
				{
					mac =  mac + MxA[k][i][j]*MxB[k][i][j];
				}

			}

	return mac;
}

//Integer MAC
template<int mtx_h, int mtx_w, int mtx_d>
int int_only_mx_mac(int (&MxA)[mtx_d][mtx_h][mtx_w], int (&MxB)[mtx_d][mtx_h][mtx_w])
{


	int i,j,k;
	int mac = 0;

	for(k=0; k<mtx_d; k++)
			for(i=0; i<mtx_h; i++)
			{
				for(j=0;j<mtx_w;j++)
				{
					mac =  mac + MxA[k][i][j]*MxB[k][i][j];
				}
			}

	return mac;
}



/////////////////////////////////////////////////////////////////////////
///---------------INTEGERIZE FUNCTION ---------------------------
////////////////////////////////////////////////////////////////////////
template<int mtx_h, int mtx_w, int mtx_d>
float integerize_M(float (&x)[mtx_d][mtx_h][mtx_w], ap_int<INT_SIZE+2> (&y)[mtx_d][mtx_h][mtx_w]) //x : input
{
	int i,j,k;
	float max = 0;
	float scalar;
	float max_c[mtx_h][mtx_w] ; //channel-wise max
	float sum=0;


	//find max(abs(x)) <-- max|x| ; x:2D/3D

		for(i=0; i<mtx_h; i++)
			for(j=0;j<mtx_w;j++)
			{
				max_c[i][j] = 0; //since max of abs is found 0 will be the min
				for(k=0; k<mtx_d; k++) //channel-wise
			{
				max = (abs_f(x[k][i][j]) > (max)) ? abs_f(x[k][i][j]) : max; //fcmp

				max_c[i][j] = (abs_f(x[k][i][j]) > (max_c[i][j])) ? abs_f(x[k][i][j]) : max_c[i][j]; //fcmp //channelwise max

			}

				sum = sum+max_c[i][j]; //fadd //sum up channel wise max matrix


			}


	//scalar = max / pow(2,INT_SIZE);//avoiding pow func
	scalar = sum / 32;


	//Integerized matrix
	for(k=0; k<mtx_d; k++)
	{
			for(i=0; i<mtx_h; i++)
			{
				for(j=0;j<mtx_w;j++)
				{
					//y[k][i][j] =round((x[k][i][j]/max)*pow(2,INT_SIZE)); //avoiding pow func
					y[k][i][j] =round((x[k][i][j]/max)*32); //fmul , fdiv
				}
			}
	}

	return scalar;

}




template<int mtx_h, int mtx_w, int mtx_d>
float integerize(float (&x)[mtx_d][mtx_h][mtx_w], ap_int<INT_SIZE+2> (&y)[mtx_d][mtx_h][mtx_w]) //x : input
{
	int i,j,k;
	float max = 0;
	float scalar;

	//find max(abs(x)) <-- max|x| ; x:2D/3D
	for(k=0; k<mtx_d; k++)
		for(i=0; i<mtx_h; i++)
			for(j=0;j<mtx_w;j++)
			{
				max = (abs_f(x[k][i][j]) > (max)) ? abs_f(x[k][i][j]) : max; //fcmp
			}


	//scalar = max / pow(2,INT_SIZE);//avoiding pow func
	scalar = max / 32;


	//Integerized matrix
	for(k=0; k<mtx_d; k++)
	{
			for(i=0; i<mtx_h; i++)
			{
				for(j=0;j<mtx_w;j++)
				{
					//y[k][i][j] =round((x[k][i][j]/max)*pow(2,INT_SIZE)); //avoiding pow func
					y[k][i][j] =round((x[k][i][j]/max)*32); //fmul , fdiv
				}
			}
	}

	return scalar;

}

//
template<int mtx_h, int mtx_w>
float integerize_2d(float (&x)[mtx_h][mtx_w], ap_int<INT_SIZE+2> (&y)[mtx_h][mtx_w]) //x : input
{
	int i,j;
	float max = 0;
	float scalar;

	//find max(abs(x)) <-- max|x| ; x:2D/3D

		integerize_2d_max:for(i=0; i<mtx_h; i++)

//#pragma HLS PIPELINE II=1
for(j=0;j<mtx_w;j++)
			{
				max = (abs_f(x[i][j]) > (max)) ? abs_f(x[i][j]) : max; //fcmp
			}


	//scalar = max / pow(2,INT_SIZE);//avoiding pow func
	scalar = max / 32;


	//Integerized matrix

	integerize_2d_convert_top: for(i=0; i<mtx_h; i++)
			{
				integerize_2d_convert: for(j=0;j<mtx_w;j++)
				{
//#pragma HLS PIPELINE II=1
					y[i][j] =round((x[i][j]/max)*32); //fmul , fdiv
				}
			}


	return scalar;

}


//
template<int mtx_h>
float integerize_1d(float (&x)[mtx_h], ap_int<INT_SIZE+2> (&y)[mtx_h]) //x : input
{
	int i;
	float max = 0;
	float scalar;

	//find max(abs(x)) <-- max|x| ; x:2D/3D

		for(i=0; i<mtx_h; i++)
			{
				max = (abs_f(x[i]) > (max)) ? abs_f(x[i]) : max; //fcmp
			}


	//scalar = max / pow(2,INT_SIZE);//avoiding pow func
	scalar = max / 32;


	//Integerized matrix
      for(i=0; i<mtx_h; i++)

				{
					y[i] =round((x[i]/max)*32); //fmul , fdiv
				}


	return scalar;

}






/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
//--------Misc functions----------
////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2, size_t mtx_d, size_t mtx_h, size_t mtx_w >
void print_mtx(T1 (&x)[mtx_d][mtx_h][mtx_w], T2 scalar )
{

			int i,j,k;

			//printf("\nPrinting Matrix\n");
			for(k=0; k<mtx_d; k++)
					{
							for(i=0; i<mtx_h; i++)
							{
								for(j=0;j<mtx_w;j++)
								{
									//if(typeid(T1)  == typeid(ap_int<4>) && typeid(T2)  == typeid(ap_int<4>))
										cout<<x[k][i][j]*scalar<<"\t";
								}
								printf("\n");
							}
							printf("\n");
					}
}


template <typename T1, typename T2, size_t mtx_h, size_t mtx_w >
void print_mtx_2d(T1 (&x)[mtx_h][mtx_w], T2 scalar )
{

			int i,j;

			//printf("\nPrinting Matrix\n");

							for(i=0; i<mtx_h; i++)
							{
								for(j=0;j<mtx_w;j++)
								{
									//if(typeid(T1)  == typeid(ap_int<4>) && typeid(T2)  == typeid(ap_int<4>))
										cout<<x[i][j]*scalar<<"\t";
								}
								printf("\n");
							}
							printf("\n");

}



template <typename T1, typename T2, size_t mtx_h >
void print_mtx_1d(T1 (&x)[mtx_h], T2 scalar )
{

			int i,j;

			//printf("\nPrinting Matrix\n");

							for(i=0; i<mtx_h; i++)
							{
										cout<<x[i]*scalar<<"\t";
							}
							printf("\n");

}


#endif
