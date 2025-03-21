#include <array>
#include <cmath>
#include <cstddef>
#include <print>
#include <span>
#include <vector>

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
    vkc::PushConstantManager pushConstantMgr{kernelSize};

    std::array<float, uboLen> weights;
    std::array<float, uboLen> writeBackWeights;
    genGaussKernel(weights, kernelSize, 1.5);
    vkc::UBOManager uboManager{phyDeviceMgr, deviceMgr, sizeof(weights)};
    vkc::SSBOManager ssboManager{phyDeviceMgr, deviceMgr, sizeof(writeBackWeights)};

    vkc::ImageManager srcImageMgr{phyDeviceMgr, deviceMgr, srcImage.getExtent(), vkc::ImageType::ReadOnly};
    vkc::ImageManager dstImageMgr{phyDeviceMgr, deviceMgr, srcImage.getExtent(), vkc::ImageType::ReadWrite};

    std::vector descPoolSizes = genPoolSizes(samplerMgr, srcImageMgr, dstImageMgr, uboManager, ssboManager);
    vkc::DescPoolManager descPoolMgr{deviceMgr, descPoolSizes};
    std::array descSetLayoutBindings =
        genDescSetLayoutBindings(samplerMgr, srcImageMgr, dstImageMgr, uboManager, ssboManager);
    vkc::DescSetLayoutManager descSetLayoutMgr{deviceMgr, descSetLayoutBindings};
    vkc::PipelineLayoutManager pipelineLayoutMgr{deviceMgr, descSetLayoutMgr, pushConstantMgr.getPushConstantRange()};
    vkc::DescSetManager descSetMgr{deviceMgr, descSetLayoutMgr, descPoolMgr};
    descSetMgr.updateDescSets(samplerMgr, srcImageMgr, dstImageMgr, uboManager, ssboManager);

    // Pipeline
    vkc::ShaderManager computeShaderMgr{deviceMgr, vkc::gaussianBlurSpirvCode};
    constexpr vkc::BlockSize blockSize{16, 16, 1};
    vkc::PipelineManager pipelineMgr{deviceMgr, pipelineLayoutMgr, computeShaderMgr};

    // Command Buffer
    vkc::QueueManager queueMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandPoolManager commandPoolMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandBufferManager commandBufferMgr{deviceMgr, commandPoolMgr};
    vkc::TimestampQueryPoolManager queryPoolMgr{deviceMgr, 2, phyDeviceMgr.getTimestampPeriod()};

    commandBufferMgr.begin();
    commandBufferMgr.bindPipeline(pipelineMgr);
    commandBufferMgr.bindDescSet(descSetMgr, pipelineLayoutMgr);
    commandBufferMgr.pushConstant(pushConstantMgr, pipelineLayoutMgr);
    commandBufferMgr.recordUpload(srcImageMgr);
    commandBufferMgr.recordLayoutTransUndefToDst(dstImageMgr);
    commandBufferMgr.recordResetQueryPool(queryPoolMgr);
    commandBufferMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eTopOfPipe);
    commandBufferMgr.recordDispatch(srcImage.getExtent(), blockSize);
    commandBufferMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eTopOfPipe);
    commandBufferMgr.recordDownload(dstImageMgr);
    commandBufferMgr.end();

    // Upload Data
    srcImageMgr.uploadFrom(srcImage.getImageSpan());
    uboManager.uploadFrom({(std::byte*)weights.data(), sizeof(weights)});

    // Actual Execution
    commandBufferMgr.submitTo(queueMgr);
    commandBufferMgr.waitFence();

    // Download Data
    ssboManager.downloadTo({(std::byte*)writeBackWeights.data(), sizeof(writeBackWeights)});
    dstImageMgr.downloadTo(dstImage.getImageSpan());
    dstImage.saveTo("out.png");

    const auto& elapsedTimes = queryPoolMgr.getElaspedTimes();
    std::println("Compute shader timecost: {} ms", elapsedTimes[0]);

    // Crosscheck SSBO
    for (int i = 0; i <= kernelSize / 2; i++) {
        assert(std::abs(writeBackWeights[i] - weights[i]) < 1e-10);
    }
}
