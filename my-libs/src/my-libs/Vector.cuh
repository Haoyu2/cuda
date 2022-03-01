//
// Created by haoyu on 2/28/2022.
//

#ifndef CUDA_DEMO_VECTOR_CUH
#define CUDA_DEMO_VECTOR_CUH


#include <cstdint>
#include <iostream>
#include <cassert>

template<typename T>
__global__ void VecAdd(T *A, T *B, T *C, uint64_t N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}


template<typename T>
class Vector {
private:
    const uint64_t len, size;
    T *data;
    int threadsPerBlock;
    uint64_t blocksPerGrid;

    void setBlocksPerGrid() {
        blocksPerGrid =
                (size + threadsPerBlock - 1) / threadsPerBlock;
    }

    void init() {
        cudaMallocManaged(&data, this->size);
        cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
        threadsPerBlock = 256;
        setBlocksPerGrid();
    }

public:
    Vector(const uint64_t len) :
            len(len),
            size(len * sizeof(T)),
            data() {
        init();
    }


    Vector<T> operator+(Vector<T> other) {
        Vector<T> res(len);
        VecAdd<<<blocksPerGrid, threadsPerBlock>>>(
                data, other.data, res.data, len
        );
        cudaDeviceSynchronize(); // important
        return res;

    }

    void setThreadsPerBlock(int threadsPerBlock) {
        Vector::threadsPerBlock = threadsPerBlock;
        blocksPerGrid =
                (size + threadsPerBlock - 1) / threadsPerBlock;
    }

    Vector setRand(uint64_t max = LLONG_MAX) {
        for (uint64_t i = 0; i < len; i++) data[i] = std::rand() % max;
        return *this;
    }

    void print() {
        std::cout << "len:" << len << ": ";
        for (uint64_t i = 0; i < len; i++) std::cout << data[i] << ' ';
        std::cout << '\n';
    }

    uint64_t getLen() const {
        return len;
    }

    uint64_t getSize() const {
        return size;
    }

    T *getData() const {
        return data;
    }
};

template<class T>
void verify_Vector_Addition(Vector<T> a, Vector<T> b, Vector<T> c) {
    for (uint64_t i = 0; i < a.getLen(); i++) {
        assert(c.getData()[i] == a.getData()[i] + b.getData()[i]);
    }
    std::cout << "Correct!\n";
}

//template<typename T>
//void verify_Vector_Addition<T, T, T>(Vector<T> a, Vector<T> b, Vector<T> c) {
//    for (uint64_t i = 0; i < a.len; i++) {
//        assert(c.data[i] == a.data[i] + b.data[i]);
//    }
//}

#endif //CUDA_DEMO_VECTOR_CUH
