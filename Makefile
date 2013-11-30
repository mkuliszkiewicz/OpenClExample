# main: main.c
# 	gcc -o main main.c -I/usr/local/cuda/include -lOpenCL
mmul: mmul.c
	gcc -o mmul mmul.c -I/usr/local/cuda/include -lOpenCL