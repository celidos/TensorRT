#include "kernel.h"

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void SeLUKernel(const int n, const float* input, float* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        float mx = input[i] > 0.0f ? input[i] : 0.0f;
	float mn = 1.673263242f * (exp(input[i]) - 1.0f);
	if (mn < 0.0f) {
	    mn = 0.0f;
	}
	output[i] = 1.05070098f * (mx + mn);
    }
}

pluginStatus_t SeLUGPU(cudaStream_t stream, const int n, const void* input, void* output)
{
    const int BS = 512;
    const int GS = (n + BS - 1) / BS;
    SeLUKernel<BS><<<GS, BS, 0, stream>>>(n,
                                           (const float*) input,
                                           (float*) output);
    return STATUS_SUCCESS;
}

pluginStatus_t SeLUInference(
    cudaStream_t stream, const int n, const void* input, void* output)
{
    return SeLUGPU(stream, n, (const float*) input, (float*) output);
}
