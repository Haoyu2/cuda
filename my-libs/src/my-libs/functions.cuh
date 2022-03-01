//
// Created by haoyu on 2/14/2022.
//

#ifndef CUDA_DEMO_FUNCTIONS_CUH
#define CUDA_DEMO_FUNCTIONS_CUH

#endif //CUDA_DEMO_FUNCTIONS_CUH

#include <cstdint>

template<typename T>
struct Vec {
    const uint8_t size;
    const uint64_t len;
    T *const nums;
};

template<typename T>
struct VecGPU {
    const uint8_t size;
    const uint64_t len;
    T *const nums;
};

template<typename T>
Vec<T> newVec(uint64_t len);

template<typename T>
Vec<T> newRandVec(uint64_t len);

template<typename T>
Vec<T> vecAdd(Vec<T> a, Vec<T> b);

template<typename T>
VecGPU<T> newVecGPU(uint64_t len);

template<typename T>
VecGPU<T> vecAddGPU(Vec<T> a, Vec<T> b);

template<typename T>
Vec<T> vecGPU2CPU(VecGPU<T> a);
template<typename T>
VecGPU<T> vecCPU2GPU(Vec<T> vecC);

template<typename T>
__global__ void _vecAddGPU(T *A, T *B, T *C, uint64_t N);


/**
 * swap two variables
 * @tparam T
 * @param a
 * @param b
 */
template<typename T>
void swap(T &a, T &b);


void demo();
//template <typename T>
//class Vector{
//private:
//    const uint64_t len;
//    const T *data;
//public:
//    explicit Vector(uint64_t len);
//    virtual ~Vector();
//
//    Vector operator+(const Vector<T> &);
//
//};
