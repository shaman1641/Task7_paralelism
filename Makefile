all:gpu
	
gpu: main_gpu.cpp
	pgc++ -acc=gpu -Minfo=all -cudalib=cublas -lboost_program_options -o gpu main_gpu.cpp

clean:gpu
	rm gpu