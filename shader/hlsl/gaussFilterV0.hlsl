struct PushConstants {
    int kernelSize;
    float sigma;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0)]] Texture2D<float4> srcTex;
[[vk::binding(1)]] SamplerState srcSampler;
[[vk::binding(2)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> dstImage;

[numthreads(16, 16, 1)] void main(uint3 globalTid : SV_DispatchThreadID) {
    const int2 dstIdx = int2(globalTid.xy);
    int2 dstSize;
    dstImage.GetDimensions(dstSize.x, dstSize.y);
    if (dstIdx.x >= dstSize.x || dstIdx.y >= dstSize.y) {
        return;
    }

    const int kSize = pc.kernelSize;
    const int halfKSize = kSize / 2;
    const float sigma = pc.sigma;

    float4 color = {0.0, 0.0, 0.0, 0.0};
    float weightSum = 0.0;
    for (int y = -halfKSize; y <= halfKSize; y++) {
        for (int x = -halfKSize; x <= halfKSize; x++) {
            const float weight = exp(-float(x * x + y * y) / (sigma * sigma * 2.0));
            const int2 inCoord = dstIdx + int2(x, y);
            const float2 uv = (float2(inCoord) + 0.5) / float2(dstSize);
            const float4 srcVal = srcTex.SampleLevel(srcSampler, uv, 0);
            color += srcVal * weight;
            weightSum += weight;
        }
    }
    color /= weightSum;

    dstImage[dstIdx] = float4(color.rgb, 1.0);
}
