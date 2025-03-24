[[vk::binding(0)]] Texture2D<float4> srcTex;
[[vk::binding(1)]] SamplerState srcSampler;
[[vk::binding(2)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> dstImage;

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int2 dstIdx = int2(dtid.xy);
    int2 dstSize;
    dstImage.GetDimensions(dstSize.x, dstSize.y);
    if (dstIdx.x >= dstSize.x || dstIdx.y >= dstSize.y)
    {
        return;
    }

    float2 uv = (float2(dstIdx) + 0.5) / float2(dstSize);
    float4 srcVal = srcTex.SampleLevel(srcSampler, uv, 0);
    float gray = 0.299 * srcVal.r + 0.587 * srcVal.g + 0.114 * srcVal.b;
    dstImage[dstIdx] = float4(gray, gray, gray, 1.0);
}
