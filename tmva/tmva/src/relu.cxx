#ifndef CUSTOM_RELU_OP_HXX
#define CUSTOM_RELU_OP_HXX

#include "onnxruntime_cxx_api.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// --- CUDA Kernel Definition ---
extern "C" {

// The CUDA kernel that computes element-wise RELU.
__global__ void relu_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

} // extern "C"

// --- Custom ONNX Operator Implementation ---

// CustomReluKernel wraps the CUDA kernel call.
struct CustomReluKernel {
    CustomReluKernel(const OrtApi& api, const OrtKernelInfo* info)
        : api_(api), info_(info) {}

    void Compute(OrtKernelContext* context) {
        // Get the input tensor.
        const OrtValue* input_tensor = api_.KernelContext_GetInput(context, 0);
        const float* input_data = api_.GetTensorData<float>(input_tensor);

        // Retrieve tensor dimensions.
        OrtTensorDimensions dims(api_, input_tensor);
        size_t total_len = 1;
        for (size_t i = 0; i < dims.size(); i++) {
            total_len *= dims[i];
        }

        // Allocate the output tensor (same shape as input).
        OrtValue* output_tensor = api_.KernelContext_GetOutput(context, 0, dims.data(), dims.size());
        float* output_data = api_.GetTensorMutableData<float>(output_tensor);

        // Allocate device memory.
        float *d_input, *d_output;
        cudaError_t err = cudaMalloc(&d_input, total_len * sizeof(float));
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to allocate device memory for input.");
        err = cudaMalloc(&d_output, total_len * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_input);
            throw std::runtime_error("Failed to allocate device memory for output.");
        }

        // Copy input data from host to device.
        err = cudaMemcpy(d_input, input_data, total_len * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("cudaMemcpy H2D failed.");
        }

        // Launch the CUDA kernel.
        int threads = 256;
        int blocks = (total_len + threads - 1) / threads;
        relu_kernel<<<blocks, threads>>>(d_input, d_output, total_len);

        // Wait for kernel to finish (optional, but recommended for error checking).
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("CUDA kernel execution failed.");
        }

        // Copy the result from device to host.
        err = cudaMemcpy(output_data, d_output, total_len * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("cudaMemcpy D2H failed.");
        }

        // Free device memory.
        cudaFree(d_input);
        cudaFree(d_output);
    }

private:
    const OrtApi& api_;
    const OrtKernelInfo* info_;
};

// CustomReluOp registers the custom operator with ONNX Runtime.
struct CustomReluOp : Ort::CustomOpBase<CustomReluOp, CustomReluKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new CustomReluKernel(api, info);
    }

    const char* GetName() const { return "CustomRelu"; }

    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

#endif // CUSTOM_RELU_OP_HXX
