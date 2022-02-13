#include "definitions.hpp"
#include "input.dat"

int main()

{

	float outd2[OUT_H_D2][OUT_W_D2];

	#void CNN_top(float (&in1)[IN_D_1][IN_H_1][IN_W_1],float (&out5)[OUT_H_D2][OUT_W_D2]);
	void CNN_top_int(float (&in1)[IN_D_1][IN_H_1][IN_W_1],float (&out5)[OUT_H_D2][OUT_W_D2]);

	
	CNN_top_int(im2_interict, outd2);
	cout<<"Network output is... \n";
			for(int i=0; i<OUT_H_D2; i++)
						for(int j=0;j<OUT_W_D2;j++)
						{
										cout<<"\n "<<outd2[i][j];
						}

}
