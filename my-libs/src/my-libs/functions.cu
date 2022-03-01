//
// Created by haoyu on 2/14/2022.
//

#include "functions.cuh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>


template<typename T>
Vec<T> vecGPU2CPU(VecGPU<T> a) {
    Vec<T> res = newVec<T>(a.len);
    cudaMemcpy(res.nums, a.nums, a.len * a.size, cudaMemcpyDeviceToHost);
    return res;
}

template<typename T>
Vec<T> newVec(uint64_t len) {
    return {sizeof(T), len, (T *) malloc(len * sizeof(T))};
}

template<typename T>
VecGPU<T> newVecGPU(uint64_t len) {
    VecGPU<T> vecG = {sizeof(T), len};
    cudaMalloc((void **) &vecG.nums, vecG.len * vecG.size);
    return vecG;
}

template<typename T>
Vec<T> newRandVec(uint64_t len) {
    Vec<T> vec = {sizeof(T), len, (T *) (malloc(len * sizeof(T)))};
    for (uint64_t i = 0; i < len; i++) vec.nums[i] = std::rand();
    return vec;
}


template<typename T>
Vec<T> vecAdd(Vec<T> a, Vec<T> b) {
    VecGPU<T> resG = vecAddGPU(vecCPU2GPU<>(a), vecCPU2GPU<T>(b));
    return vecGPU2CPU(resG);
}

template<typename T>
VecGPU<T> vecCPU2GPU(Vec<T> vecC) {
    VecGPU<T> vecG = {sizeof(T), vecC.len};
    cudaMalloc((void **) &vecG.nums, vecG.len * vecG.size);
    cudaMemcpy(vecG.nums, vecC.nums, vecG.len * vecG.size, cudaMemcpyDeviceToDevice);
    return vecG;
}

template<typename T>
VecGPU<T> vecAddGPU(VecGPU<T> a, VecGPU<T> b) {
    VecGPU<T> res = newVecGPU<T>(a.len);
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (a.len + threadsPerBlock - 1) / threadsPerBlock;
    _vecAddGPU<<<blocksPerGrid, threadsPerBlock>>>(a.nums, b.nums, res.nums, a.len);
    return res;
}

template<typename T>
__global__ void _vecAddGPU(T *A, T *B, T *C, uint64_t N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

template<typename T>
void verifyVecAddition(Vec<T> a, Vec<T> b, Vec<T> c) {
    for (uint64_t i = 0; i < a.len; i++) {
        std:: cout << c.nums[i] << " "
        << a.nums[i] << " " << b.nums[i] << "\n";
//        assert(c.nums[i] == a.nums[i] + b.nums[i]);
    }
}
template<typename T>
void print(Vec<T> a){
    for (uint64_t i = 0; i < a.len; i++) {
        std:: cout << a.nums[i] << " "                   ;

    }
    std:: cout << "\n";
}

void demo() {
    int a = 1, b = 2;
    swap<int>(a, b);
    std::cout << a << ' ' << b << "\n";
    int N = 10;
    Vec<int> vec1 = newRandVec<int>(N), vec2 = newRandVec<int>(N);
    print<int>(vec1);
    print<int>(vec2);
    Vec<int> res = vecAdd(vec1, vec2);
    verifyVecAddition(vec1, vec2, res);


}


// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < N) c[tid] = a[tid] + b[tid];
}


// Check vector add result
void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
    for (int i = 0; i < a.size(); i++) {

//        std:: cout << c[i] << a[i] + b[i];
        assert(c[i] == a[i] + b[i]);
    }
}

// Kernel definition
__global__ void MatAdd(float **A,
                       float **B,
                       float **C) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}


template<typename T>
void swap(T &a, T &b) {
    T tmp{a};
    a = b;
    b = tmp;
}


using namespace std;

void vertorAdd_demo() {
    // Array size of 2^16 (65536 elements)
    constexpr int N = 1 << 4;
    constexpr size_t bytes = sizeof(int) * N;

    // Vectors for holding the host-side (CPU-side) data
    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c;
    c.reserve(N);

    // Initialize random numbers in each array
    for (int i = 0; i < N; i++) {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
        std::cout << a[i] << ' ' << b[i];
    }

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA (1024)
    int NUM_THREADS = 1 << 10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous (the CPU program continues execution after
    // call, but no necessarily before the kernel finishes)
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    // Copy sum vector from device to host
    // cudaMemcpy is a synchronous operation, and waits for the prior kernel
    // launch to complete (both go to the default stream in this case).
    // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
    // barrier.
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
    verify_result(a, b, c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "COMPLETED SUCCESSFULLY\n";

}