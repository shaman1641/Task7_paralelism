#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "cublas_v2.h"
namespace opt = boost::program_options;

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

void initMatrix(std::unique_ptr<double[]> &arr ,int N){
    arr[0] = 10.0;
    arr[N-1] = 20.0;
    arr[(N-1)*N + (N-1)] = 30.0;
    arr[(N-1)*N] = 20.0;
    for (size_t i = 1; i < N-1; i++){
        arr[0*N+i] = linearInterpolation(i,0.0,arr[0],N-1,arr[N-1]);
        arr[i*N+0] = linearInterpolation(i,0.0,arr[0],N-1,arr[(N-1)*N]);
        arr[i*N+(N-1)] = linearInterpolation(i,0.0,arr[N-1],N-1,arr[(N-1)*N + (N-1)]);
        arr[(N-1)*N+i] = linearInterpolation(i,0.0,arr[(N-1)*N],N-1,arr[(N-1)*N + (N-1)]);
    }
}


int main(int argc, char const *argv[]) {
    opt::options_description desc("опции");
    desc.add_options()
        ("accuracy",opt::value<double>()->default_value(1e-6),"точность")
        ("cellsCount",opt::value<int>()->default_value(256),"размер матрицы")
        ("iterCount",opt::value<int>()->default_value(1000000),"количество операций")
    ;
    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    int N = vm["cellsCount"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int countIter = vm["iterCount"].as<int>();

    std::unique_ptr<double[]> A(new double[N*N]);
    std::unique_ptr<double[]> Anew(new double[N*N]);
    initMatrix(std::ref(A),N);
    initMatrix(std::ref(Anew),N);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double* curmatrix = A.get();
    double* prevmatrix = Anew.get();
    double error = 1.0;
    int iter = 0;
    double alpha = -1.0;
    int idx=0;
   
    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc data copyin(idx,prevmatrix[0:N*N],curmatrix[0:N*N],N,alpha)
    {
        while (iter < countIter && iter<10000000 && error > accuracy){
            #pragma acc parallel loop independent collapse(2) vector vector_length(1024) gang num_gangs(1024) present(curmatrix,prevmatrix)
                for (size_t i = 1; i < N-1; i++){
                    for (size_t j = 1; j < N-1; j++){
                        curmatrix[i*N+j]  =(prevmatrix[i*N+j+1] + prevmatrix[i*N+j-1] + prevmatrix[(i-1)*N+j] + prevmatrix[(i+1)*N+j]) * 0.25;
                    }
                }

                if ((iter+1)%100 == 0){
                    #pragma acc data present(prevmatrix,curmatrix) wait
                    #pragma acc host_data use_device(curmatrix,prevmatrix)
                    {
                        cublasDaxpy(handle,N*N,&alpha,curmatrix,1,prevmatrix,1);
                        cublasIdamax(handle,N*N,prevmatrix,1,&idx);
                    }
                    #pragma acc update self(prevmatrix[idx-1])
                    error = fabs(prevmatrix[idx-1]);
                    #pragma acc host_data use_device(curmatrix,prevmatrix)
                    {
                        cublasDcopy(handle,N*N,curmatrix,1,prevmatrix,1);
                    }
                
                }
                        
                double* temp = prevmatrix;
                prevmatrix = curmatrix;
                curmatrix = temp;
                iter++;

        }
    
        cublasDestroy(handle);
        #pragma acc update self(curmatrix[0:N*N])
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double time_s = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())/1000;
    std::cout<<"time: " << time_s<<" error: "<<error << " iterarion: " << iter<<std::endl;
    
    if (N ==13 || N == 10){
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                std::cout << A[i*N+j] << ' ';  
            }
            std::cout << std::endl;
        }
    }

    A = nullptr;
    Anew = nullptr;

    return 0;
}