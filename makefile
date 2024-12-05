all: conv

conv: cuDconv.cu makefile
	nvcc -lineinfo --use_fast_math --fmad=true -arch=native -o cuDconv cuDconv.cu --library-path /usr/local/cuda/targets/x86_64-linux/lib/stubs --library cuda -Xptxas -v --maxrregcount 170
