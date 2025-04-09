struct PushConstants {
    int kernelSize;
    float sigma2;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0)]] Texture2D<float4> srcTex;
[[vk::binding(1)]] SamplerState srcSampler;
[[vk::binding(2)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> dstImage;

static const int MAX_HALF_KSIZE = 128;
static const int GROUP_SIZE = 256;
static const int SHARED_MEM_SIZE = GROUP_SIZE + 2 * MAX_HALF_KSIZE;
// Gathered Y for each X
groupshared float4 gatheredY[SHARED_MEM_SIZE];
// Each thread should sample multiple times to fill up the `gatheredY`
static const int ALIGNED_SHARED_MEM_SIZE = (SHARED_MEM_SIZE + (GROUP_SIZE - 1)) & ((~GROUP_SIZE) + 1);
static const int SAMPLE_TIMES = ALIGNED_SHARED_MEM_SIZE / GROUP_SIZE;

[numthreads(GROUP_SIZE, 1, 1)] void main(uint3 dispTid : SV_DispatchThreadID, uint3 groupID : SV_GroupID,
                                         uint3 groupTID : SV_GroupThreadID) {
    const int2 dstIdx = int2(dispTid.xy);
    int2 dstSize;
    dstImage.GetDimensions(dstSize.x, dstSize.y);
    const float2 invDstSize = 1.0 / float2(dstSize);
    const int halfKSize = pc.kernelSize / 2;

    const int srcStartX = groupID.x * GROUP_SIZE + groupTID.x - MAX_HALF_KSIZE;
    const int srcStartY = groupID.y;

    // Gather col cache for each `[srcStartX, srcStartX + GROUP_SIZE, srcStartX + 2*GROUP_SIZE, ...]` cols
    for (int x = 0; x < SAMPLE_TIMES; x++) {
        const int smemWriteX = GROUP_SIZE * x + groupTID.x;
        if (smemWriteX < SHARED_MEM_SIZE) {
            // Gather from `[srcStartY-halfKSize, srcStartY+halfKSise]` rows
            const int2 iUv = int2(srcStartX + GROUP_SIZE * x, srcStartY - halfKSize);
            const float2 uv = (float2(iUv) + 0.5) * invDstSize;
            float accWeight = exp(-float(halfKSize * halfKSize) / pc.sigma2);
            float4 acc = srcTex.SampleLevel(srcSampler, uv, 0) * accWeight;
            for (int y = 1 - halfKSize; y <= halfKSize; y += 2) {
                const int negY2 = -y * y;
                const float weightUp = exp(float(negY2) / pc.sigma2);
                const float weightDown = exp(float(negY2 - (y << 1) - 1) / pc.sigma2);
                const float weight = weightUp + weightDown;
                const float yOffset = (float(y + halfKSize) + weightDown / weight) * invDstSize.y;
                const float4 srcVal = srcTex.SampleLevel(srcSampler, float2(uv.x, uv.y + yOffset), 0);
                acc = mad(srcVal, weight, acc);
                accWeight += weight;
            }
            gatheredY[smemWriteX] = acc / accWeight;
        }
    }

    // Never put the barrier after the divergence !!!
    GroupMemoryBarrierWithGroupSync();

    if (dstIdx.x >= dstSize.x || dstIdx.y >= dstSize.y) {
        return;
    }

    // Calculate the final output by gathering cached cols
    float4 acc = float4(0.0, 0.0, 0.0, 0.0);
    float accWeight = 0.0;
    for (int x = -halfKSize; x <= halfKSize; x++) {
        const float weight = exp(-float(x * x) / pc.sigma2);
        const int smemX = MAX_HALF_KSIZE + groupTID.x + x;
        const float4 srcVal = gatheredY[smemX];
        acc = mad(srcVal, weight, acc);
        accWeight += weight;
    }
    const float4 dstVal = acc / accWeight;

    dstImage[dstIdx] = float4(dstVal.rgb, 1.0);
}
