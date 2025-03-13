struct PushConstants {
    int kernelSize;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0)]] SamplerState srcSampler;
[[vk::binding(1)]] Texture2D<float4> srcTex;
[[vk::binding(2)]] RWTexture2D<float4> dstImage;

struct UBO {
    float4 weights[4];
};
[[vk::binding(3)]] ConstantBuffer<UBO> ubo;

[[vk::binding(4)]] RWStructuredBuffer<float> writeBackWeights;


float4 blurAtIdx(Texture2D tex, int2 idx)
{
    int kSize = pc.kernelSize;
    int halfKSize = kSize / 2;
    int2 srcSize;
    tex.GetDimensions(srcSize.x, srcSize.y);

    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    for (int y = -halfKSize; y <= halfKSize; y++)
    {
        float4 rowAcc = float4(0.0, 0.0, 0.0, 0.0);

        for (int x = -halfKSize; x <= halfKSize; x++)
        {
            int2 inCoord = idx + int2(x, y);
            float2 uv = (float2(inCoord) + 0.5) / float2(srcSize);
            float4 srcVal = tex.SampleLevel(srcSampler, uv, 0);

            int absX = abs(x);
            int weightY = absX >> 2;
            int weightX = absX & (4 - 1);
            float weight = ubo.weights[weightY][weightX];
            rowAcc += srcVal * weight;
        }

        int absY = abs(y);
        int weightY = absY >> 2;
        int weightX = absY & (4 - 1);
        float weight = ubo.weights[weightY][weightX];
        color += rowAcc * weight;
    }

    return color;
}

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int2 dstIdx = int2(dtid.xy);

    int kSize = pc.kernelSize;
    int halfKSize = kSize / 2;
    if (dstIdx.x <= halfKSize)
    {
        int weightY = dstIdx.x >> 2;
        int weightX = dstIdx.x & (4 - 1);
        writeBackWeights[dstIdx.x] = ubo.weights[weightY][weightX];
    }

    int2 dstSize;
    dstImage.GetDimensions(dstSize.x, dstSize.y);
    if (dstIdx.x >= dstSize.x || dstIdx.y >= dstSize.y)
    {
        return;
    }

    float4 color = blurAtIdx(srcTex, dstIdx);

    dstImage[dstIdx] = float4(color.rgb, 1.0);
}
