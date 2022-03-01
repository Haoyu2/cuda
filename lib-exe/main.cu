#include <iostream>
#include <math.h>
#include <my-libs/functions.cuh>
#include <my-libs/Vector.cuh>
#include <my-libs/Matrix.cuh>


void test1() {
    Vector<int> a(10), b(10), res(10);
    a.setRand(10).print();
    b.setRand(10).print();
    (a + b).print();
}

void test0() {
    int N = 100000;
    Vector<int> a(N), b(N);
    a.setRand();
    b.setRand();
    verify_Vector_Addition<>(a, b, a + b);

}


void test2() {
    int M = 10, K = 10, N = 10;
    Matrix<int> A(M, K), B(K, N);

    A.setRand(10).print();
    B.setRand(10).print();
    A.multiply(B).print();
    A.multiply(B).print();
    verify_Matrix_MultiplicationNaive(A, B, A.multiply(B));
}

void test3() {
    int M = 10, K = 10, N = 10;
    Matrix<int> A(M, K), B(K, N);

    A.setRand(10).print();
    B.setRand(10).print();

    verify_Matrix_MatMinMaxKernelNaive(A, B, A.minmax(B));
}

void test4() {
    int M = 10, K = 10, N = 10;
    Matrix<int> A(M, K), B(K, N);

    A.setRand(10).print();
    B.setRand(10).print();

    A.minmax(B).print();
    A.minmaxCPU(B).print();
}

void running_time_gpu(int NUM = 10) {

    for (int i = 5; i < NUM; i++) {
        int M = 1 << i;
        Matrix<int> A(M, M), B(M, M);
        A.setRand();
        B.setRand();

//        std::cout << M << ": ";
        clock_t tStart = clock();
        Matrix<int> C = A.minmax(B);
        std::cout << (clock() - tStart) << " ";
        A.free();
        B.free();
        C.free();
    }
}

void running_time_cpu(int NUM = 10) {

    for (int i = 5; i < NUM; i++) {
        int M = 1 << i;
        Matrix<int> A(M, M), B(M, M);
        A.setRand();
        B.setRand();
        clock_t tStart = clock();

        Matrix<int> C = A.minmaxCPU(B);
        std::cout << (clock() - tStart) << " ";
        A.free();
        B.free();
        C.free();

    }
}

int main(void) {

//    demo();
//    test4();
    int NUM = 10;
    running_time_gpu(NUM);
    std::cout << "\n";
    running_time_cpu(NUM);

    return 0;
}