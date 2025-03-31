struct PushConstants {
    int kernelSize;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0)]] Texture2D<float4> srcTex;
[[vk::binding(1)]] SamplerState srcSampler;
[[vk::binding(2)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> dstImage;

struct UBO {
    float4 weights[4];
};
[[vk::binding(3)]] ConstantBuffer<UBO> ubo;

[numthreads(16, 16, 1)] void main(uint3 dtid : SV_DispatchThreadID) {
    int2 dstIdx = int2(dtid.xy);
    int2 dstSize;
    dstImage.GetDimensions(dstSize.x, dstSize.y);
    if (dstIdx.x >= dstSize.x || dstIdx.y >= dstSize.y) {
        return;
    }

    int kSize = pc.kernelSize;
    int halfKSize = kSize / 2;

    float4 color = {0.0, 0.0, 0.0, 0.0};
    for (int y = -halfKSize; y <= halfKSize; y++) {
        float4 rowAcc = {0.0, 0.0, 0.0, 0.0};

        for (int x = -halfKSize; x <= halfKSize; x++) {
            int2 inCoord = dstIdx + int2(x, y);
            float2 uv = (float2(inCoord) + 0.5) / float2(dstSize);
            float4 srcVal = srcTex.SampleLevel(srcSampler, uv, 0);

            int absX = abs(x);
            int weightY = absX >> 2;
            int weightX = absX & (4 - 1);
            float weight = ubo.weights[weightY][weightX];
            rowAcc = mad(srcVal, weight, rowAcc);
        }

        int absY = abs(y);
        int weightY = absY >> 2;
        int weightX = absY & (4 - 1);
        float weight = ubo.weights[weightY][weightX];
        color = mad(rowAcc, weight, color);
    }

    dstImage[dstIdx] = float4(color.rgb, 1.0);
}
