struct PushConstants {
    int kernelSize;
    float sigma2;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0)]] Texture2D<float4> srcTex;
[[vk::binding(1)]] SamplerState srcSampler;
[[vk::binding(2)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> dstImage;

[numthreads(16, 16, 1)] void main(uint3 dispTid : SV_DispatchThreadID) {
    const int2 dstIdx = int2(dispTid.xy);
    int2 dstSize;
    dstImage.GetDimensions(dstSize.x, dstSize.y);
    if (dstIdx.x >= dstSize.x || dstIdx.y >= dstSize.y) {
        return;
    }

    const float2 invDstSize = 1.0 / float2(dstSize);
    const int halfKSize = pc.kernelSize / 2;

    float4 color = {0.0, 0.0, 0.0, 0.0};
    float weightSum = 0.0;
    for (int y = -halfKSize; y <= halfKSize; y++) {
        for (int x = -halfKSize; x <= halfKSize; x++) {
            const int2 offset = int2(x, y);
            const int2 inCoord = dstIdx + int2(x, y);
            const float2 uv = (float2(inCoord) + 0.5) * invDstSize;
            const float4 srcVal = srcTex.SampleLevel(srcSampler, uv, 0);
            const float weight = exp(-float(dot(offset, offset)) / pc.sigma2);
            color = mad(srcVal, weight, color);
            weightSum += weight;
        }
    }
    color /= weightSum;

    dstImage[dstIdx] = float4(color.rgb, 1.0);
}
