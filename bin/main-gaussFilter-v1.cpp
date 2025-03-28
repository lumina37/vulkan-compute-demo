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
    vkc::ImageManager dstImageMgr{phyDeviceMgr, deviceMgr, srcImage.getExtent(), vkc::ImageType::Write};

    std::vector descPoolSizes = genPoolSizes(srcImageMgr, samplerMgr, dstImageMgr, gaussKernelWeightsMgr);
    vkc::DescPoolManager descPoolMgr{deviceMgr, descPoolSizes};

    std::array gaussDLayoutBindings =
        genDescSetLayoutBindings(srcImageMgr, samplerMgr, dstImageMgr, gaussKernelWeightsMgr);
    vkc::DescSetLayoutManager gaussDLayoutMgr{deviceMgr, gaussDLayoutBindings};
    vkc::PipelineLayoutManager gaussPLayoutMgr{deviceMgr, gaussDLayoutMgr, kernelSizePcMgr.getPushConstantRange()};
    vkc::DescSetManager gaussDescSetMgr{deviceMgr, gaussDLayoutMgr, descPoolMgr};
    gaussDescSetMgr.updateDescSets(srcImageMgr, samplerMgr, dstImageMgr, gaussKernelWeightsMgr);

    // Pipeline
    constexpr vkc::BlockSize blockSize{16, 16, 1};
    vkc::ShaderManager gaussShaderMgr{deviceMgr, shader::gaussFilterV1SpirvCode};
    vkc::PipelineManager gaussPipelineMgr{deviceMgr, gaussPLayoutMgr, gaussShaderMgr};

    // Command Buffer
    vkc::QueueManager queueMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandPoolManager commandPoolMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandBufferManager gaussCmdBufMgr{deviceMgr, commandPoolMgr};
    vkc::TimestampQueryPoolManager queryPoolMgr{deviceMgr, 2, phyDeviceMgr.getTimestampPeriod()};

    // Gaussian Blur
    srcImageMgr.uploadFrom(srcImage.getImageSpan());
    gaussKernelWeightsMgr.uploadFrom({(std::byte*)gaussKernelWeights.data(), sizeof(gaussKernelWeights)});

    for (int i = 0; i < 15; i++) {
        gaussCmdBufMgr.begin();
        gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
        gaussCmdBufMgr.bindDescSet(gaussDescSetMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.recordResetQueryPool(queryPoolMgr);
        gaussCmdBufMgr.recordSrcPrepareTranfer(srcImageMgr);
        gaussCmdBufMgr.recordUploadToSrc(srcImageMgr);
        gaussCmdBufMgr.recordSrcPrepareShaderRead(srcImageMgr);
        gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgr);
        gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader);
        gaussCmdBufMgr.recordDispatch(srcImage.getExtent(), blockSize);
        gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader);
        gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgr);
        gaussCmdBufMgr.recordDownloadToDst(dstImageMgr);
        gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgr);
        gaussCmdBufMgr.end();

        gaussCmdBufMgr.submitTo(queueMgr);
        gaussCmdBufMgr.waitFence();

        std::println("Gaussian blur timecost: {} ms", queryPoolMgr.getElaspedTimes()[0]);
    }

    dstImageMgr.downloadTo(dstImage.getImageSpan());
    dstImage.saveTo("out.png");
}
