# HLS_IntegerNet
FPGA based hardware acceleration for Seizure detection CNN using IntegerNet Algorithm[1] implemented in HLS on PYNQ-Z2 board.

This consists of C++ templates required to build CNN based on IntegerNet Algorithm also its benchmark floating-point model.

_Note_ : Code is for int-5 model and int5 folder consists of the example model parameters and input.dat has the sample preprocessed inputs.

## Observation:
Improvement in latency and on-chip memory consumption. 

## Reference
<a id = "1">[1]</a>
N. D. Truong et al., "Integer Convolutional Neural Network for Seizure Detection," in IEEE Journal on Emerging and Selected Topics in Circuits and Systems, vol. 8, no. 4, pp. 849-857, Dec. 2018, doi: 10.1109/JETCAS.2018.2842761.
