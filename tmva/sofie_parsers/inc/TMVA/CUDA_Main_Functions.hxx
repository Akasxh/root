#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// ReLU kernel: output[i] = (input[i] > 0) ? input[i] : 0
template <typename T>
__global__ void reluKernel(const T* input, T* output, size_t length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
       output[id] = (input[id] > T(0)) ? input[id] : T(0);
    }
}

// Leaky ReLU kernel: output[i] = (input[i] > 0) ? input[i] : alpha * input[i]
template <typename T>
__global__ void leakyReluKernel(const T* input, T* output, size_t length, T alpha) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
       output[id] = (input[id] > T(0)) ? input[id] : alpha * input[id];
    }
}

// ELU kernel: output[i] = (input[i] >= 0) ? input[i] : alpha * (exp(input[i]) - 1)
template <typename T>
__global__ void eluKernel(const T* input, T* output, size_t length, T alpha) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
       T inVal = input[id];
       output[id] = (inVal >= T(0)) ? inVal : alpha * (exp(inVal) - T(1));
    }
}

// Tanh kernel: output[i] = tanh(input[i])
template <typename T>
__global__ void tanhKernel(const T* input, T* output, size_t length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
       output[id] = tanhf(input[id]);
    }
}

// GEMM kernel: C = A * B
// A: MxK, B: KxN, C: MxN
template <typename T>
__global__ void gemmKernel(const T* A, const T* B, T* C, int M, int N, int K) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   if (row < M && col < N) {
      T sum = 0;
      for (int k = 0; k < K; ++k) {
         sum += A[row * K + k] * B[k * N + col];
      }
      C[row * N + col] = sum;
   }
}


void testReLU() {
    std::cout << "Testing ReLU Kernel\n";
    const size_t length = 5;
    float h_input[length] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f};
    float h_output[length] = {0};

    float *d_input, *d_output;
    cudaMalloc(&d_input, length * sizeof(float));
    cudaMalloc(&d_output, length * sizeof(float));
    cudaMemcpy(d_input, h_input, length * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    reluKernel<float><<<blocks, threadsPerBlock>>>(d_input, d_output, length);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, length * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "ReLU Output: ";
    for (size_t i = 0; i < length; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n\n";

    cudaFree(d_input);
    cudaFree(d_output);
}

void testLeakyReLU() {
    std::cout << "Testing LeakyReLU Kernel\n";
    const size_t length = 5;
    float h_input[length] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f};
    float h_output[length] = {0};

    float *d_input, *d_output;
    cudaMalloc(&d_input, length * sizeof(float));
    cudaMalloc(&d_output, length * sizeof(float));
    cudaMemcpy(d_input, h_input, length * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 0.1f;
    int threadsPerBlock = 256;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    leakyReluKernel<float><<<blocks, threadsPerBlock>>>(d_input, d_output, length, alpha);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, length * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "LeakyReLU Output (alpha = " << alpha << "): ";
    for (size_t i = 0; i < length; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n\n";

    cudaFree(d_input);
    cudaFree(d_output);
}

void testELU() {
    std::cout << "Testing ELU Kernel\n";
    const size_t length = 5;
    float h_input[length] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f};
    float h_output[length] = {0};

    float *d_input, *d_output;
    cudaMalloc(&d_input, length * sizeof(float));
    cudaMalloc(&d_output, length * sizeof(float));
    cudaMemcpy(d_input, h_input, length * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    int threadsPerBlock = 256;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    eluKernel<float><<<blocks, threadsPerBlock>>>(d_input, d_output, length, alpha);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, length * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "ELU Output (alpha = " << alpha << "): ";
    for (size_t i = 0; i < length; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n\n";

    cudaFree(d_input);
    cudaFree(d_output);
}

void testTanh() {
    std::cout << "Testing Tanh Kernel\n";
    const size_t length = 5;
    float h_input[length] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f};
    float h_output[length] = {0};

    float *d_input, *d_output;
    cudaMalloc(&d_input, length * sizeof(float));
    cudaMalloc(&d_output, length * sizeof(float));
    cudaMemcpy(d_input, h_input, length * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    tanhKernel<float><<<blocks, threadsPerBlock>>>(d_input, d_output, length);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, length * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Tanh Output: ";
    for (size_t i = 0; i < length; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n\n";

    cudaFree(d_input);
    cudaFree(d_output);
}

void testGEMM() {
    std::cout << "Testing GEMM Kernel\n";
    // Matrix dimensions: A (MxK), B (KxN), C (MxN)
    int M = 2, K = 3, N = 2;
    // Define host matrices
    // A is 2x3
    float h_A[] = {1, 2, 3, 4, 5, 6};
    // B is 3x2
    float h_B[] = {7, 8, 9, 10, 11, 12};
    // C is 2x2 (result)
    float h_C[4] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions for 2D kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    gemmKernel<float><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "GEMM Output (Matrix C):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    testReLU();
    testLeakyReLU();
    testELU();
    testTanh();
    testGEMM();

    return 0;
}
