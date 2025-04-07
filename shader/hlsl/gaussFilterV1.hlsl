struct PushConstants {
    int kernelSize;
    float sigma;
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
    const int halfKSize = pc.kernelSize / 2;
    const float sigma2 = pc.sigma * pc.sigma * 2.0;

    const int srcStartX = groupID.x * GROUP_SIZE + groupTID.x - MAX_HALF_KSIZE;
    const int srcStartY = groupID.y;
    const int2 srcBaseCoord = int2(srcStartX, srcStartY);

    // Gather col cache for each `[srcStartX, srcStartX + GROUP_SIZE, srcStartX + 2*GROUP_SIZE, ...]` cols
    for (int x = 0; x < SAMPLE_TIMES; x++) {
        const int smemWriteX = GROUP_SIZE * x + groupTID.x;
        if (smemWriteX < SHARED_MEM_SIZE) {
            // Gather from `[srcStartY-halfKSize, srcStartY+halfKSise]` rows
            float4 acc = float4(0.0, 0.0, 0.0, 0.0);
            float accWeight = 0.0;
            const float inX = (float(srcStartX + GROUP_SIZE * x) + 0.5) / float(dstSize.x);
            for (int y = -halfKSize; y <= halfKSize; y++) {
                const float inY = (float(srcStartY + y) + 0.5) / float(dstSize.y);
                const float4 srcVal = srcTex.SampleLevel(srcSampler, float2(inX, inY), 0);
                const float weight = exp(-float(y * y) / sigma2);
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
        const float weight = exp(-float(x * x) / sigma2);
        const int smemX = MAX_HALF_KSIZE + groupTID.x + x;
        const float4 srcVal = gatheredY[smemX];
        acc = mad(srcVal, weight, acc);
        accWeight += weight;
    }
    const float4 dstVal = acc / accWeight;

    dstImage[dstIdx] = float4(dstVal.rgb, 1.0);
}
