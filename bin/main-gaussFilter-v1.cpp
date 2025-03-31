#include <array>
#include <cmath>
#include <cstddef>
#include <print>
#include <span>
#include <vector>

#include "spirv/gaussFilter.hpp"
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
    vkc::PhysicalDeviceManager phyDeviceMgr{instMgr};
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceMgr);
    auto pDeviceMgr = std::make_shared<vkc::DeviceManager>(phyDeviceMgr, computeQFamilyIdx);
    vkc::QueueManager queueMgr{*pDeviceMgr, computeQFamilyIdx};

    // Descriptor & Layouts
    vkc::SamplerManager samplerMgr{pDeviceMgr};

    constexpr int uboLen = 16;
    constexpr int maxKernelSize = uboLen * 2 + 1;
    constexpr int kernelSize = 11;
    static_assert(kernelSize <= maxKernelSize);
    vkc::PushConstantManager kernelSizePcMgr{kernelSize};

    std::array<float, uboLen> gaussKernelWeights;
    genGaussKernel(gaussKernelWeights, kernelSize, 1.5);
    vkc::UBOManager gaussKernelWeightsMgr{phyDeviceMgr, pDeviceMgr, sizeof(gaussKernelWeights)};

    vkc::ImageManager srcImageMgr{phyDeviceMgr, pDeviceMgr, srcImage.getExtent(), vkc::ImageType::Read};
    std::array srcImageMgrCRefs{std::cref(srcImageMgr)};
    vkc::ImageManager dstImageMgr{phyDeviceMgr, pDeviceMgr, srcImage.getExtent(), vkc::ImageType::Write};
    std::array dstImageMgrCRefs{std::cref(dstImageMgr)};

    std::vector descPoolSizes = genPoolSizes(srcImageMgr, samplerMgr, dstImageMgr, gaussKernelWeightsMgr);
    vkc::DescPoolManager descPoolMgr{pDeviceMgr, descPoolSizes};

    std::array gaussDLayoutBindings =
        genDescSetLayoutBindings(srcImageMgr, samplerMgr, dstImageMgr, gaussKernelWeightsMgr);
    vkc::DescSetLayoutManager gaussDLayoutMgr{pDeviceMgr, gaussDLayoutBindings};
    vkc::PipelineLayoutManager gaussPLayoutMgr{pDeviceMgr, gaussDLayoutMgr, kernelSizePcMgr.getPushConstantRange()};
    vkc::DescSetManager gaussDescSetMgr{pDeviceMgr, gaussDLayoutMgr, descPoolMgr};
    gaussDescSetMgr.updateDescSets(srcImageMgr, samplerMgr, dstImageMgr, gaussKernelWeightsMgr);

    // Pipeline
    constexpr vkc::BlockSize blockSize{16, 16, 1};
    vkc::ShaderManager gaussShaderMgr{pDeviceMgr, shader::gaussFilterV1SpirvCode};
    vkc::PipelineManager gaussPipelineMgr{pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr};

    // Command Buffer
    vkc::CommandPoolManager commandPoolMgr{pDeviceMgr, computeQFamilyIdx};
    vkc::CommandBufferManager gaussCmdBufMgr{pDeviceMgr, commandPoolMgr};
    vkc::TimestampQueryPoolManager queryPoolMgr{pDeviceMgr, 2, phyDeviceMgr.getTimestampPeriod()};

    // Gaussian Blur
    srcImageMgr.uploadFrom(srcImage.getImageSpan());
    gaussKernelWeightsMgr.uploadFrom({(std::byte*)gaussKernelWeights.data(), sizeof(gaussKernelWeights)});

    for (int i = 0; i < 15; i++) {
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

        gaussCmdBufMgr.submitTo(queueMgr);
        gaussCmdBufMgr.waitFence();

        std::println("Gaussian blur timecost: {} ms", queryPoolMgr.getElaspedTimes()[0]);
    }

    dstImageMgr.downloadTo(dstImage.getImageSpan());
    dstImage.saveTo("out.png");
}
