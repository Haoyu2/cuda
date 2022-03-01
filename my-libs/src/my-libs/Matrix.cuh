//
// Created by haoyu on 2/28/2022.
//

#ifndef CUDA_DEMO_MATRIX_CUH
#define CUDA_DEMO_MATRIX_CUH

#include <cassert>

#include <cstdint>
#include <iostream>

template<typename T>
__global__ void MatMulKernelNaive(
        T *A, T *B, T *C, int M, int K, int N, int SIZE) {
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    int ic = row * N + col, ra = row * K;
    C[ic] = 0;// important
    if (row < M && col < N) { // important
        for (int i = 0; i < K; i++) {
            // Accumulate results for a single element
            C[ic] += A[ra + i] * B[i * N + col];
        }
    }

}

template<typename T>
__global__ void MatMinMaxKernelNaive(
        T *A, T *B, T *C, int M, int K, int N, int SIZE) {
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    int ic = row * N + col, ra = row * K;

    if (row < M && col < N) { // important
        C[ic] = A[ra + 0] < B[col] ? B[col] : A[ra + 0];
        for (int i = 1; i < K; i++) {
            // Accumulate results for a single element
            int tmp = A[ra + i] < B[i * N + col] ? B[i * N + col] : A[ra + i];
            if (tmp < C[ic]) C[ic] = tmp;
        }
    }

}

template<typename T>
class Matrix {
private:
    const u_int32_t ROWS, COLS;
    const u_int64_t SIZE;
    T *data;

    void init() {
        cudaMallocManaged(&data, SIZE);
        cudaMemAdvise(data, SIZE, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    }

public:

    Matrix(const u_int32_t rows, const u_int32_t cols) :
            ROWS(rows), COLS(cols), SIZE((u_int64_t) ROWS * COLS), data() {
        init();
    }

    Matrix setRand(uint64_t max = LLONG_MAX) {
        for (uint64_t i = 0; i < SIZE; i++) data[i] = std::rand() % max;
        return *this;
    }

    Matrix setZero() {
        memset(data, 0, SIZE);
        return *this;
    }


    Matrix multiply(Matrix<T> other, int THREADS = 32) {
        Matrix<T> res(ROWS, other.COLS);
//        res.setZero();
        // Blocks per grid dimension (assumes THREADS divides M and N evenly)
        int BLOCKS_X = ROWS / THREADS + 1;
        int BLOCKS_Y = other.COLS / THREADS + 1;

        // Use dim3 structs for block  and grid dimensions
        dim3 threads(THREADS, THREADS);
        dim3 blocks(BLOCKS_X, BLOCKS_Y);

        MatMinMaxKernelNaive<<<blocks, threads>>>(
                data, other.data, res.data, ROWS, COLS, other.COLS, res.SIZE
        );
        cudaDeviceSynchronize(); // important
        return res;

    }

    Matrix minmax(Matrix<T> other, int THREADS = 32) {
        Matrix<T> res(ROWS, other.COLS);
        // Blocks per grid dimension (assumes THREADS divides M and N evenly)
        int BLOCKS_X = ROWS / THREADS + 1;
        int BLOCKS_Y = other.COLS / THREADS + 1;

        // Use dim3 structs for block  and grid dimensions
        dim3 threads(THREADS, THREADS);
        dim3 blocks(BLOCKS_X, BLOCKS_Y);

        MatMinMaxKernelNaive<<<blocks, threads>>>(
                data, other.data, res.data, ROWS, COLS, other.COLS, res.SIZE
        );
        cudaDeviceSynchronize(); // important
        return res;

    }



    Matrix minmaxCPU(Matrix<T> other) {

        Matrix<T> res(ROWS, other.COLS);
//        std::cout << res.COLS << " ";

        for (uint64_t i = 0; i < ROWS; i++) {
            for (uint64_t j = 0; j < other.getCols(); j++) {
                T cur = max(get(i, 0), other.get(0, j));
                for (int k = 1; k < COLS; k++) {
                    cur = min(cur, max(get(i, k), other.get(k, j)));
                }
                res.set(i, j, cur);
            }
        }
        return res;
    }


    u_int32_t getRows() const {
        return ROWS;
    }

    u_int32_t getCols() const {
        return COLS;
    }

    u_int64_t getSize() const {
        return SIZE;
    }

    T *getData() const {
        return data;
    }

    T get(int i, int j) {
        return data[i * COLS + j];
    }
    T set(int i, int j, T val) {
        return data[i * COLS + j] = val;
    }
    void print() {
        std::cout << "DIM:" << ROWS << " * " << COLS << '\n';
        for (uint64_t i = 0; i < ROWS; i++) {
            for (uint64_t j = 0; j < COLS; j++)
                std::cout << data[i * COLS + j] << ' ';
            std::cout << '\n';
        }
        std::cout << '\n';
    }
    void free(){
        cudaFree(data);
    }
};

template<class T>
void verify_Matrix_MatMinMaxKernelNaive(Matrix<T> a, Matrix<T> b, Matrix<T> c) {
    c.print();
    for (uint64_t i = 0; i < a.getRows(); i++) {
        for (uint64_t j = 0; j < b.getCols(); j++) {
            int cur = max(a.get(i, 0), b.get(0, j));
            for (int k = 1; k < a.getCols(); k++) {
                cur = min(cur, max(a.get(i, k), b.get(k, j)));
            }
            std::cout << c.get(i, j) << "=="
                      << cur << "\n";
            assert(c.get(i, j) == cur);

        }
    }
    std::cout << "Correct!\n";
}


template<class T>
void verify_Matrix_MultiplicationNaive(Matrix<T> a, Matrix<T> b, Matrix<T> c) {
    c.print();
    for (uint64_t i = 0; i < a.getRows(); i++) {
        for (uint64_t j = 0; j < b.getCols(); j++) {
            long long cur = 0;
            for (int k = 0; k < a.getCols(); k++) {
                cur += a.get(i, k) * b.get(k, j);
            }
//            std::cout << c.get(i, j) << "=="
//                      << cur << "\n";
            assert(c.get(i, j) == cur);

        }
    }
    std::cout << "Correct!\n";
}


#endif //CUDA_DEMO_MATRIX_CUH
