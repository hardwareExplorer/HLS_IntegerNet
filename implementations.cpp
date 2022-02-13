#ifndef IMPL_TOP
#define IMPL_TOP
#include "definitions.hpp"


//--------------------------NETWORK_IMPLMENTATION---------------------------------------------------
float outc1[OUT_D_1][OUT_H_1][OUT_W_1];
float outp1[OUT_D_P1][OUT_H_P1][OUT_W_P1];
float outc2[OUT_D_2][OUT_H_2][OUT_W_2];
float outp2[OUT_D_P2][OUT_H_P2][OUT_W_P2];
float outc3[OUT_D_3][OUT_H_3][OUT_W_3];
float outp3[OUT_D_P3][OUT_H_P3][OUT_W_P3];
float outflt[IN_H_D1][IN_W_D1];
float outd1[OUT_H_D1][OUT_W_D1];
//float outd2[OUT_H_D2][OUT_W_D2];


void CNN_top(float (&in1)[IN_D_1][IN_H_1][IN_W_1],float (&out5)[OUT_H_D2][OUT_W_D2])
{
 //layer1
	BN_real(in1, mean_1_real, variance_1_real, beta_1_real, gamma_1_real);
	conv_real(in1, wt1,STRIDE_V_1, STRIDE_H_1, outc1);
	maxpool_real(outc1, K_H_P1,K_W_P1, STRIDE_V_P1, STRIDE_H_P1, outp1);

//layer2
	BN_real(outp1, mean_2_real, variance_2_real, beta_2_real, gamma_2_real);
	conv_real(outp1, wt2,STRIDE_V_2, STRIDE_H_2, outc2);
	maxpool_real(outc2, K_H_P2,K_W_P2, STRIDE_V_P2, STRIDE_H_P2, outp2);

//layer3
	BN_real(outp2, mean_3_real, variance_3_real, beta_3_real, gamma_3_real);
	conv_real(outp2, wt3,STRIDE_V_3, STRIDE_H_3, outc3);
	maxpool_real(outc3, K_H_P3, K_W_P3, STRIDE_V_P3, STRIDE_H_P3, outp3);

	flatten(outp3, outflt);

//layer4
	dense_top_real(outflt, wt4,  outd1, ACT1);

//layer5
	dense_top_real(outd1, wt5, out5 , ACT2);

}


void CNN_top_int(float (&in1)[IN_D_1][IN_H_1][IN_W_1],float (&out5)[OUT_H_D2][OUT_W_D2])
{

//layer1
	BN_real(in1, mean_1_int, variance_1_int, beta_1_int, gamma_1_int);
	conv_int(in1, scalar1, wt1_int,STRIDE_V_1, STRIDE_H_1, outc1);
	maxpool_real(outc1, K_H_P1,K_W_P1, STRIDE_V_P1, STRIDE_H_P1, outp1);

//layer2
	BN_real(outp1, mean_2_int, variance_2_int, beta_2_int, gamma_2_int);
	conv_int(outp1, scalar2, wt2_int,STRIDE_V_2, STRIDE_H_2, outc2);
	maxpool_real(outc2, K_H_P2,K_W_P2, STRIDE_V_P2, STRIDE_H_P2, outp2);

//layer3
	BN_real(outp2, mean_3_int, variance_3_int, beta_3_int, gamma_3_int);
	conv_int(outp2, scalar3, wt3_int,STRIDE_V_3, STRIDE_H_3, outc3);
	maxpool_real(outc3, K_H_P3, K_W_P3, STRIDE_V_P3, STRIDE_H_P3, outp3);

	flatten(outp3, outflt);

//layer4
	dense_top_int(outflt, scalar4, wt4_int,  outd1, ACT1);

//layer5
	dense_top_int(outd1, scalar5, wt5_int, out5 , ACT2);


}


void CNN_top_hybrid(float (&in1)[IN_D_1][IN_H_1][IN_W_1],float (&out5)[OUT_H_D2][OUT_W_D2])
{

//layer1
	BN_real(in1, mean_1_int, variance_1_int, beta_1_int, gamma_1_int);
	conv_int(in1, scalar1, wt1_int,STRIDE_V_1, STRIDE_H_1, outc1);
	maxpool_real(outc1, K_H_P1,K_W_P1, STRIDE_V_P1, STRIDE_H_P1, outp1);

//layer2
	BN_real(outp1, mean_2_int, variance_2_int, beta_2_int, gamma_2_int);
	conv_int(outp1, scalar2, wt2_int,STRIDE_V_2, STRIDE_H_2, outc2);
	maxpool_real(outc2, K_H_P2,K_W_P2, STRIDE_V_P2, STRIDE_H_P2, outp2);

//layer3
	BN_real(outp2, mean_3_int, variance_3_int, beta_3_int, gamma_3_int);
	conv_int(outp2, scalar3, wt3_int,STRIDE_V_3, STRIDE_H_3, outc3);
	maxpool_real(outc3, K_H_P3, K_W_P3, STRIDE_V_P3, STRIDE_H_P3, outp3);

	flatten(outp3, outflt);

//layer4
	dense_top_int(outflt, scalar4, wt4_int,  outd1, ACT1);

//layer5
	dense_top_real(outd1, wt5, out5 , ACT2);


}

void top_CNN_real(hls::stream<AXI_STREAM>(&in_stream), hls::stream<AXI_STREAM> (&out_stream) )
{
//#pragma HLS DATAFLOW
#pragma HLS INTERFACE ap_ctrl_none port=return //bundle=CONTROL_BUS
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE axis port=in_stream


	//<type, insize, h, w, outsize, u, ti, td>

	AXI_STREAM a,b;
	converter_f flt;
	float in1[IN_D_1][IN_H_1][IN_W_1];
	float outd2[OUT_H_D2][OUT_W_D2];

		//assert(sizeof(datatype)*8 == 32);

		//1. stream in the input matrix
		for(int k=0; k<IN_D_1; k++)
			for(int i=0; i<IN_H_1; i++)
				for(int j=0; j<IN_W_1; j++)
			{
				in_stream.read(a);
				flt.uidata = a.data;
				in1[k][i][j] = flt.fdata;
			}


		//2. Perform matrix manipulation

		CNN_top(in1, outd2);


		//3. Stream out the matrix
		for(int i=0; i<OUT_H_D2; i++)
				for(int j=0; j<OUT_W_D2; j++)
				{
					flt.fdata = outd2[i][j];
					b.data = flt.uidata;
					if(i == OUT_W_D2-1 && j== OUT_W_D2-1)
						b.last = 1;
					else
						b.last = 0;

					b.keep = 0xf;
					b.strb = 0xf;

					//other side channels
							//b.user = a.user;
							//b.id = a.id;
							//b.dest = a.dest;
					out_stream.write(b);

				}


}




void top_CNN_int(hls::stream<AXI_STREAM>(&in_stream), hls::stream<AXI_STREAM> (&out_stream) )
{
//#pragma HLS DATAFLOW
#pragma HLS INTERFACE ap_ctrl_none port=return //bundle=CONTROL_BUS
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE axis port=in_stream


	//<type, insize, h, w, outsize, u, ti, td>

	AXI_STREAM a,b;
	converter_f flt;
	float in1[IN_D_1][IN_H_1][IN_W_1];
	float outd2[OUT_H_D2][OUT_W_D2];

		//assert(sizeof(datatype)*8 == 32);

		//1. stream in the input matrix
		for(int k=0; k<IN_D_1; k++)
			for(int i=0; i<IN_H_1; i++)
				for(int j=0; j<IN_W_1; j++)
			{
				in_stream.read(a);
				flt.uidata = a.data;
				in1[k][i][j] = flt.fdata;
			}


		//2. Perform matrix manipulation

		CNN_top_int(in1, outd2);


		//3. Stream out the matrix
		for(int i=0; i<OUT_H_D2; i++)
				for(int j=0; j<OUT_W_D2; j++)
				{
					flt.fdata = outd2[i][j];
					b.data = flt.uidata;
					if(i == OUT_W_D2-1 && j== OUT_W_D2-1)
						b.last = 1;
					else
						b.last = 0;

					b.keep = 0xf;
					b.strb = 0xf;

					//other side channels
							//b.user = a.user;
							//b.id = a.id;
							//b.dest = a.dest;
					out_stream.write(b);

				}


}
void top_CNN_hybrid(hls::stream<AXI_STREAM>(&in_stream), hls::stream<AXI_STREAM> (&out_stream) )
{
//#pragma HLS DATAFLOW
#pragma HLS INTERFACE ap_ctrl_none port=return //bundle=CONTROL_BUS
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE axis port=in_stream


	//<type, insize, h, w, outsize, u, ti, td>

	AXI_STREAM a,b;
	converter_f flt;
	float in1[IN_D_1][IN_H_1][IN_W_1];
	float outd2[OUT_H_D2][OUT_W_D2];

		//assert(sizeof(datatype)*8 == 32);

		//1. stream in the input matrix
		for(int k=0; k<IN_D_1; k++)
			for(int i=0; i<IN_H_1; i++)
				for(int j=0; j<IN_W_1; j++)
			{
				in_stream.read(a);
				flt.uidata = a.data;
				in1[k][i][j] = flt.fdata;
			}


		//2. Perform matrix manipulation

		CNN_top_hybrid(in1, outd2);


		//3. Stream out the matrix
		for(int i=0; i<OUT_H_D2; i++)
				for(int j=0; j<OUT_W_D2; j++)
				{
					flt.fdata = outd2[i][j];
					b.data = flt.uidata;
					if(i == OUT_W_D2-1 && j== OUT_W_D2-1)
						b.last = 1;
					else
						b.last = 0;

					b.keep = 0xf;
					b.strb = 0xf;

					//other side channels
							//b.user = a.user;
							//b.id = a.id;
							//b.dest = a.dest;
					out_stream.write(b);

				}


}


#endif


