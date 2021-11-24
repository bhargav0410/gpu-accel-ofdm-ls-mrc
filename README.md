# gpu-accel-ofdm-ls-mrc
GPU and CPU codes for implementing OFDM Least Squares channel estimation and demodulation. Tested and implemented on ORBIT Testbed.

This repository consists of all files used for implementation of Least Sqaures channel estimation and Maximal Ratio Combining based demodulation for OFDM symbols.
Implementation and testing has been done using the USRP B210 and X310 SDRs, Intel Xeon CPU E5-2630 v3 CPU and Nvidia Tesla K40m GPU of ORBIT Testbed at WINLAB, Rutgers University.

The results based on using the code in the repository have been published in the following conference paper:
B. Gokalgandhi, C. Segerholm, N. Paul and I. Seskar, "Accelerating Channel Estimation and Demodulation of Uplink OFDM symbols for Large Scale Antenna Systems using GPU," 2019 International Conference on Computing, Networking and Communications (ICNC), 2019, pp. 955-959, doi: 10.1109/ICCNC.2019.8685544.
