// Description: Test the CUDA-based RELU operator.

 #include <iostream>
 #include <vector>
 #include <cassert>
 #include <cmath>
 #include <cuda_runtime.h>
 
 // Include the CUDA operator header.
 #include "TMVA/ROperator_Relu_CUDA.hxx"
 #include "TMVA/RModel.hxx"
 
 #include <map>
 #include <memory>
 #include <stdexcept>
 
 // Dummy implementation of RModel to support operator testing.
 class DummyRModel : public TMVA::RModel {
 public:
    std::map<std::string, std::vector<size_t>> tensorShapes;
    std::map<std::string, std::string> tensorTypes;
    std::map<std::string, std::shared_ptr<void>> tensorData;
    DummyRModel() { fUseSession = false; }
 
    bool CheckIfTensorAlreadyExist(const std::string& name) override {
       return tensorShapes.find(name) != tensorShapes.end();
    }
 
    std::vector<size_t> GetTensorShape(const std::string& name) override {
       if (!CheckIfTensorAlreadyExist(name))
          throw std::runtime_error("Tensor " + name + " not found.");
       return tensorShapes[name];
    }
 
    std::string GetTensorType(const std::string& name) override {
       if (tensorTypes.find(name) == tensorTypes.end())
          throw std::runtime_error("Tensor type for " + name + " not found.");
       return tensorTypes[name];
    }
 
    void AddIntermediateTensor(const std::string& name, const std::string& type,
                                 const std::vector<size_t>& shape) override {
       tensorShapes[name] = shape;
       tensorTypes[name] = type;
       size_t length = 1;
       for (auto s : shape) length *= s;
       float* data = new float[length];
       for (size_t i = 0; i < length; i++)
          data[i] = 0.f;
       tensorData[name] = std::shared_ptr<void>(data, std::default_delete<float[]>());
    }
 
    std::shared_ptr<void> GetInitializedTensorData(const std::string& name) override {
       if (tensorData.find(name) == tensorData.end())
          throw std::runtime_error("Tensor data for " + name + " not found.");
       return tensorData[name];
    }
 
    void UpdateInitializedTensor(const std::string& name, const std::string& type,
                                 const std::vector<size_t>& shape,
                                 std::shared_ptr<void> data) override {
       tensorShapes[name] = shape;
       tensorTypes[name] = type;
       tensorData[name] = data;
    }
 };
 
 // Helper function to compute total number of elements from a shape vector.
 size_t ConvertShapeToLength(const std::vector<size_t>& shape) {
    size_t length = 1;
    for (auto dim : shape) {
       length *= dim;
    }
    return length;
 }
 
 int main() {
    std::cout << "==== Testing CUDA-based ROperator_Relu_CUDA ====" << std::endl;
 
    // Create a simple 1D input tensor with 10 elements.
    std::vector<size_t> shapeInput = {10};
    std::vector<float> inputData = { -3.0f, 0.0f, 2.5f, -1.2f, 4.3f,
                                      -0.1f, 0.0f, 7.7f, -5.5f, 1.1f };
 
    DummyRModel model;
    std::string inputTensorName = "input";
    std::string outputTensorName = "output";
 
    // Register input tensor in the dummy model.
    model.tensorShapes[inputTensorName] = shapeInput;
    model.tensorTypes[inputTensorName] = "float";
    size_t totalElements = ConvertShapeToLength(shapeInput);
    float* inputPtr = new float[totalElements];
    for (size_t i = 0; i < totalElements; i++) {
       inputPtr[i] = inputData[i];
    }
    model.tensorData[inputTensorName] = std::shared_ptr<void>(inputPtr, std::default_delete<float[]>());
 
    // Instantiate the CUDA RELU operator.
    TMVA::Experimental::SOFIE::ROperator_Relu_CUDA<float> reluOp(inputTensorName, outputTensorName);
    // Check that shape inference works.
    std::vector<std::vector<size_t>> inferredShapes = reluOp.ShapeInference({shapeInput});
    assert(inferredShapes[0] == shapeInput);
    // Initialize operator (this registers the output tensor).
    reluOp.Initialize(model);
 
    // Retrieve the output tensor pointer.
    std::shared_ptr<void> outputDataVoid = model.tensorData[outputTensorName];
    float* outputData = static_cast<float*>(outputDataVoid.get());
 
    // --- Execute CUDA Kernel Directly ---
    float *d_input, *d_output;
    cudaMalloc(&d_input, totalElements * sizeof(float));
    cudaMalloc(&d_output, totalElements * sizeof(float));
    cudaMemcpy(d_input, inputData.data(), totalElements * sizeof(float), cudaMemcpyHostToDevice);
 
    int threads = 256;
    int blocks = (totalElements + threads - 1) / threads;
 
    // The kernel is defined in relu_kernel.cu (linkage via extern "C")
    extern __global__ void relu_kernel(const float*, float*, size_t);
    relu_kernel<<<blocks, threads>>>(d_input, d_output, totalElements);
 
    cudaMemcpy(outputData, d_output, totalElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
 
    // Validate the results.
    std::vector<float> expectedOutput;
    for (size_t i = 0; i < totalElements; i++) {
       expectedOutput.push_back(inputData[i] > 0.0f ? inputData[i] : 0.0f);
    }
 
    bool passed = true;
    for (size_t i = 0; i < totalElements; i++) {
       if (std::fabs(outputData[i] - expectedOutput[i]) > 1e-6) {
          passed = false;
          std::cout << "Mismatch at index " << i << ": expected " << expectedOutput[i]
                    << ", got " << outputData[i] << std::endl;
       }
    }
 
    if (passed)
       std::cout << "CUDA RELU operator test PASSED." << std::endl;
    else
       std::cout << "CUDA RELU operator test FAILED." << std::endl;
 
    return passed ? 0 : -1;
 }
 