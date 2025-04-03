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

    int kSize = pc.kernelSize;
    int halfKSize = kSize / 2;

    const float sigma = pc.sigma;
    float weights[16];
    weights[0] = 1.0;
    float weightSum = 1.0;
    for (int i = 1; i < halfKSize; i++) {
        const float weight = exp(-float(i * i) / (sigma * sigma * 2.0));
        weights[i] = weight;
        weightSum += weight;
    }

    float4 color = {0.0, 0.0, 0.0, 0.0};
    for (int y = -halfKSize; y <= halfKSize; y++) {
        float4 rowAcc = {0.0, 0.0, 0.0, 0.0};

        for (int x = -halfKSize; x <= halfKSize; x++) {
            int2 inCoord = dstIdx + int2(x, y);
            float2 uv = (float2(inCoord) + 0.5) / float2(dstSize);
            float4 srcVal = srcTex.SampleLevel(srcSampler, uv, 0);
            const float weight = weights[abs(x)];
            rowAcc += srcVal * weight;
        }

        const float weight = weights[abs(y)];
        color += rowAcc * weight;
    }
    color /= weightSum;

    dstImage[dstIdx] = float4(color.rgb, 1.0);
}
