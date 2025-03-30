#include <array>
#include <cmath>
#include <cstddef>
#include <print>
#include <span>
#include <vector>

#include "spirv/gaussFilter.hpp"
#include "spirv/grayscale.hpp"
#include "vkc.hpp"

void genGaussKernel(std::span<float> dst, const int kernelSize, const float sigma) {
    const int halfKSize = kernelSize / 2;
    const float doubleSigma2 = 2 * sigma * sigma;

    dst[0] = 1.;
    float sum = dst[0];
    for (int i = 1; i <= halfKSize; i++) {
        const float elem = std::expf((float)(-i * i) / doubleSigma2);
        dst[i] = elem;
        sum += 2 * elem;  // double for both side
    }
    for (auto& elem : dst) {
        elem /= sum;
    }
}

int main(int argc, char** argv) {
    vkc::StbImageManager srcImage{"in.png"};
    vkc::StbImageManager dstImage{srcImage.getExtent()};

    // Device
    vkc::InstanceManager instMgr;
    vkc::PhyDeviceManager phyDeviceMgr{instMgr};
    vkc::QueueFamilyManager queueFamilyMgr{phyDeviceMgr};
    vkc::DeviceManager deviceMgr{phyDeviceMgr, queueFamilyMgr};

    // Descriptor & Layouts
    vkc::SamplerManager samplerMgr{deviceMgr};

    constexpr int uboLen = 16;
    constexpr int maxKernelSize = uboLen * 2 + 1;
    constexpr int kernelSize = 11;
    static_assert(kernelSize <= maxKernelSize);
    vkc::PushConstantManager kernelSizePcMgr{kernelSize};

    std::array<float, uboLen> gaussKernelWeights;
    genGaussKernel(gaussKernelWeights, kernelSize, 1.5);
    vkc::UBOManager gaussKernelWeightsMgr{phyDeviceMgr, deviceMgr, sizeof(gaussKernelWeights)};

    vkc::ImageManager srcImageMgr{phyDeviceMgr, deviceMgr, srcImage.getExtent(), vkc::ImageType::Read};
    std::array srcImageMgrCRefs{std::cref(srcImageMgr)};
    vkc::ImageManager dstImageMgr{phyDeviceMgr, deviceMgr, srcImage.getExtent(), vkc::ImageType::Write};
    std::array dstImageMgrCRefs{std::cref(dstImageMgr)};
    std::array<vkc::CommandBufferManager::TImageManagerRefPair, 1> imageMgrCRefPairs{
        std::array{std::cref(srcImageMgr), std::cref(dstImageMgr)}};

    std::vector descPoolSizes =
        genPoolSizes(srcImageMgr, samplerMgr, dstImageMgr, gaussKernelWeightsMgr, srcImageMgr, samplerMgr, dstImageMgr);
    vkc::DescPoolManager descPoolMgr{deviceMgr, descPoolSizes};

    std::array gaussDLayoutBindings =
        genDescSetLayoutBindings(srcImageMgr, samplerMgr, dstImageMgr, gaussKernelWeightsMgr);
    vkc::DescSetLayoutManager gaussDLayoutMgr{deviceMgr, gaussDLayoutBindings};
    vkc::PipelineLayoutManager gaussPLayoutMgr{deviceMgr, gaussDLayoutMgr, kernelSizePcMgr.getPushConstantRange()};
    vkc::DescSetManager gaussDescSetMgr{deviceMgr, gaussDLayoutMgr, descPoolMgr};
    gaussDescSetMgr.updateDescSets(srcImageMgr, samplerMgr, dstImageMgr, gaussKernelWeightsMgr);

    std::array grayDLayoutBindings = genDescSetLayoutBindings(srcImageMgr, samplerMgr, dstImageMgr);
    vkc::DescSetLayoutManager grayDLayoutMgr{deviceMgr, grayDLayoutBindings};
    vkc::PipelineLayoutManager grayPLayoutMgr{deviceMgr, grayDLayoutMgr};
    vkc::DescSetManager grayDescSetMgr{deviceMgr, grayDLayoutMgr, descPoolMgr};
    grayDescSetMgr.updateDescSets(srcImageMgr, samplerMgr, dstImageMgr);

    // Pipeline
    constexpr vkc::BlockSize blockSize{16, 16, 1};
    vkc::ShaderManager gaussShaderMgr{deviceMgr, shader::gaussFilterV1SpirvCode};
    vkc::PipelineManager gaussPipelineMgr{deviceMgr, gaussPLayoutMgr, gaussShaderMgr};
    vkc::ShaderManager grayShaderMgr{deviceMgr, shader::grayscaleSpirvCode};
    vkc::PipelineManager grayPipelineMgr{deviceMgr, grayPLayoutMgr, grayShaderMgr};

    // Command Buffer
    vkc::QueueManager queueMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandPoolManager commandPoolMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandBufferManager gaussCmdBufMgr{deviceMgr, commandPoolMgr};
    vkc::CommandBufferManager grayCmdBufMgr{deviceMgr, commandPoolMgr};
    vkc::TimestampQueryPoolManager queryPoolMgr{deviceMgr, 2, phyDeviceMgr.getTimestampPeriod()};

    // Gaussian Blur
    gaussCmdBufMgr.begin();
    gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
    gaussCmdBufMgr.bindDescSet(gaussDescSetMgr, gaussPLayoutMgr);
    gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
    gaussCmdBufMgr.recordResetQueryPool(queryPoolMgr);
    gaussCmdBufMgr.recordSrcPrepareTranfer(srcImageMgrCRefs);
    gaussCmdBufMgr.recordUploadToSrc(srcImageMgrCRefs);
    gaussCmdBufMgr.recordSrcPrepareShaderRead(srcImageMgrCRefs);
    gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrCRefs);
    gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader);
    gaussCmdBufMgr.recordDispatch(srcImage.getExtent(), blockSize);
    gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader);
    gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgrCRefs);
    gaussCmdBufMgr.recordDownloadToDst(dstImageMgrCRefs);
    gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgrCRefs);
    gaussCmdBufMgr.end();

    srcImageMgr.uploadFrom(srcImage.getImageSpan());
    gaussKernelWeightsMgr.uploadFrom({(std::byte*)gaussKernelWeights.data(), sizeof(gaussKernelWeights)});

    gaussCmdBufMgr.submitTo(queueMgr);
    gaussCmdBufMgr.waitFence();

    dstImageMgr.downloadTo(dstImage.getImageSpan());

    std::println("Gaussian blur timecost: {} ms", queryPoolMgr.getElaspedTimes()[0]);

    // Grayscale
    grayCmdBufMgr.begin();
    grayCmdBufMgr.bindPipeline(grayPipelineMgr);
    grayCmdBufMgr.bindDescSet(grayDescSetMgr, grayPLayoutMgr);
    grayCmdBufMgr.recordResetQueryPool(queryPoolMgr);
    grayCmdBufMgr.recordSrcPrepareTranfer(srcImageMgrCRefs);
    grayCmdBufMgr.recordImageCopy(imageMgrCRefPairs);
    grayCmdBufMgr.recordSrcPrepareShaderRead(srcImageMgrCRefs);
    grayCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrCRefs);
    grayCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader);
    grayCmdBufMgr.recordDispatch(srcImage.getExtent(), blockSize);
    grayCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader);
    grayCmdBufMgr.recordDstPrepareTransfer(dstImageMgrCRefs);
    grayCmdBufMgr.recordDownloadToDst(dstImageMgrCRefs);
    grayCmdBufMgr.recordWaitDownloadComplete(dstImageMgrCRefs);
    grayCmdBufMgr.end();

    grayCmdBufMgr.submitTo(queueMgr);
    grayCmdBufMgr.waitFence();

    dstImageMgr.downloadTo(dstImage.getImageSpan());
    dstImage.saveTo("out.png");

    std::println("Grayscale timecost: {} ms", queryPoolMgr.getElaspedTimes()[0]);
}
