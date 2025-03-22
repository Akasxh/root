#ifndef CUSTOM_GEMM_OP_HXX
#define CUSTOM_GEMM_OP_HXX

#include "onnxruntime_cxx_api.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// --- CUDA Kernel Definition for GEMM ---
// This simple GEMM kernel computes C = A * B for 2D matrices.
// A is of shape (M, K), B is of shape (K, N), and C is of shape (M, N).
extern "C" {

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Each thread computes one element of matrix C.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

} // extern "C"

// CustomGemmKernel wraps the CUDA kernel call.
struct CustomGemmKernel {
    CustomGemmKernel(const OrtApi& api, const OrtKernelInfo* info)
        : api_(api), info_(info) {}

    void Compute(OrtKernelContext* context) {
        // Get input tensor A.
        const OrtValue* A_tensor = api_.KernelContext_GetInput(context, 0);
        const float* A_data = api_.GetTensorData<float>(A_tensor);
        // Get input tensor B.
        const OrtValue* B_tensor = api_.KernelContext_GetInput(context, 1);
        const float* B_data = api_.GetTensorData<float>(B_tensor);

        // Retrieve dimensions for A and B.
        OrtTensorDimensions dimsA(api_, A_tensor);
        OrtTensorDimensions dimsB(api_, B_tensor);
        if (dimsA.size() != 2 || dimsB.size() != 2)
            throw std::runtime_error("CustomGemmOp expects both inputs to be 2D matrices.");

        int M = dimsA[0];
        int K = dimsA[1];
        int K2 = dimsB[0];
        int N = dimsB[1];
        if (K != K2)
            throw std::runtime_error("Dimension mismatch: A's columns must equal B's rows.");

        // Prepare output tensor C with shape (M, N).
        std::vector<int64_t> dimsC = { M, N };
        OrtValue* C_tensor = api_.KernelContext_GetOutput(context, 0, dimsC.data(), dimsC.size());
        float* C_data = api_.GetTensorMutableData<float>(C_tensor);

        size_t sizeA = static_cast<size_t>(M * K);
        size_t sizeB = static_cast<size_t>(K * N);
        size_t sizeC = static_cast<size_t>(M * N);

        // Allocate GPU memory.
        float *d_A, *d_B, *d_C;
        cudaError_t err = cudaMalloc(&d_A, sizeA * sizeof(float));
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to allocate device memory for A.");
        err = cudaMalloc(&d_B, sizeB * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_A);
            throw std::runtime_error("Failed to allocate device memory for B.");
        }
        err = cudaMalloc(&d_C, sizeC * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_A);
            cudaFree(d_B);
            throw std::runtime_error("Failed to allocate device memory for C.");
        }

        // Copy input matrices from host to device.
        err = cudaMemcpy(d_A, A_data, sizeA * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpy H2D failed for A.");
        err = cudaMemcpy(d_B, B_data, sizeB * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpy H2D failed for B.");

        // Launch GEMM kernel.
        dim3 threads(16, 16);
        dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
        gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA kernel execution failed in GEMM.");

        // Copy result from device to host.
        err = cudaMemcpy(C_data, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpy D2H failed for C.");

        // Free device memory.
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

private:
    const OrtApi& api_;
    const OrtKernelInfo* info_;
};

// CustomGemmOp registers the custom operator with ONNX Runtime.
struct CustomGemmOp : Ort::CustomOpBase<CustomGemmOp, CustomGemmKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new CustomGemmKernel(api, info);
    }
    const char* GetName() const { return "CustomGemm"; }
    size_t GetInputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

#endif
