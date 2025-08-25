#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <vector>

#include "../vkc_helper.hpp"
#include "shader.hpp"
#include "vkc.hpp"

int main() {
    vkc::initVulkan() | unwrap;

    vkc::StbImageBox srcImage = vkc::StbImageBox::createFromPath("in.png") | unwrap;
    vkc::StbImageBox dstImage = vkc::StbImageBox::createWithExtent(srcImage.getExtent()) | unwrap;

    // Device
    vkc::DefaultInstanceProps instProps = vkc::DefaultInstanceProps::create() | unwrap;
    if (!instProps.layers.has("VK_LAYER_KHRONOS_validation")) {
        std::println(std::cerr, "VK_LAYER_KHRONOS_validation not supported");
        return -1;
    }
    vkc::InstanceBox instBox = vkc::InstanceBox::create() | unwrap;
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(instBox) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::create(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}) | unwrap);
    vkc::QueueBox queueBox = vkc::QueueBox::create(*pDeviceBox, vk::QueueFlagBits::eCompute) | unwrap;

    // Descriptor & Layouts
    vkc::SamplerBox samplerBox = vkc::SamplerBox::create(pDeviceBox) | unwrap;

    constexpr int kernelSize = 23;
    constexpr float sigma = 10.0f;
    vkc::PushConstantBox kernelSizePcBox{std::pair{kernelSize, sigma * sigma * 2.0f}};

    vkc::SampledImageBox srcImageBox =
        vkc::SampledImageBox::create(phyDeviceBox, pDeviceBox, srcImage.getExtent()) | unwrap;
    const std::array srcImageBoxRefs{std::ref(srcImageBox)};
    vkc::StorageImageBox dstImageBox =
        vkc::StorageImageBox::create(phyDeviceBox, pDeviceBox, srcImage.getExtent()) | unwrap;
    const std::array dstImageBoxRefs{std::ref(dstImageBox)};

    Timer uploadTimer;
    uploadTimer.begin();
    srcImageBox.upload(srcImage.getPData()) | unwrap;
    uploadTimer.end();
    std::println("Upload to staging timecost: {} ms", uploadTimer.durationMs());

    const std::vector descPoolSizes = genPoolSizes(srcImageBox, samplerBox, dstImageBox);
    vkc::DescPoolBox descPoolBox = vkc::DescPoolBox::create(pDeviceBox, descPoolSizes) | unwrap;

    const std::array gaussDLayoutBindings = genDescSetLayoutBindings(srcImageBox, samplerBox, dstImageBox);
    vkc::DescSetLayoutBox gaussDLayoutBox = vkc::DescSetLayoutBox::create(pDeviceBox, gaussDLayoutBindings) | unwrap;
    const std::array gaussDLayoutBoxCRefs{std::cref(gaussDLayoutBox)};
    vkc::PipelineLayoutBox gaussPLayoutBox =
        vkc::PipelineLayoutBox::createWithPushConstant(pDeviceBox, gaussDLayoutBoxCRefs,
                                                       kernelSizePcBox.getPushConstantRange()) |
        unwrap;
    vkc::DescSetsBox gaussDescSetsBox =
        vkc::DescSetsBox::create(pDeviceBox, descPoolBox, gaussDLayoutBoxCRefs) | unwrap;
    const std::array gaussWriteDescSets = genWriteDescSets(srcImageBox, samplerBox, dstImageBox);
    const std::array gaussWriteDescSetss{std::span{gaussWriteDescSets.begin(), gaussWriteDescSets.end()}};
    gaussDescSetsBox.updateDescSets(gaussWriteDescSetss);

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox gaussCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;
    vkc::TimestampQueryPoolBox queryPoolBox =
        vkc::TimestampQueryPoolBox::create(pDeviceBox, 6, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
        unwrap;

    // Pipeline
    constexpr int groupSizeX = 256;
    const int groupNumX = vkc::ceilDiv(dstImage.getExtent().width(), groupSizeX);
    const int groupNumY = dstImage.getExtent().height();
    vkc::ShaderBox gaussShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::gaussFilter::v1::code) | unwrap;
    constexpr int maxHalfKSize = 128;
    vkc::SpecConstantBox specConstantBox{groupSizeX, maxHalfKSize};
    vkc::PipelineBox gaussPipelineBox =
        vkc::PipelineBox::createCompute(pDeviceBox, gaussPLayoutBox, gaussShaderBox, specConstantBox.getSpecInfo()) |
        unwrap;

    // Record Command Buffer
    for (int i = 0; i < 15; i++) {
        gaussCmdBufBox.begin() | unwrap;
        gaussCmdBufBox.bindPipeline(gaussPipelineBox);
        gaussCmdBufBox.bindDescSets(gaussDescSetsBox, gaussPLayoutBox, vk::PipelineBindPoint::eCompute);
        gaussCmdBufBox.pushConstant(kernelSizePcBox, gaussPLayoutBox);
        gaussCmdBufBox.recordResetQueryPool(queryPoolBox);
        gaussCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::SampledImageBox>(srcImageBoxRefs);
        gaussCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufBox.recordCopyStagingToSrc(srcImageBox);
        gaussCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufBox.recordSrcPrepareShaderRead<vkc::SampledImageBox>(srcImageBoxRefs);
        gaussCmdBufBox.recordDstPrepareShaderWrite(dstImageBoxRefs);
        gaussCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        gaussCmdBufBox.recordDispatch(groupNumX, groupNumY);
        gaussCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        gaussCmdBufBox.recordPrepareSendAfterDispatch(dstImageBoxRefs);
        gaussCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufBox.recordCopyDstToStaging(dstImageBox);
        gaussCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufBox.recordWaitDownloadComplete(dstImageBoxRefs);
        gaussCmdBufBox.end() | unwrap;

        queueBox.submit(gaussCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        auto elapsedTime = queryPoolBox.getElaspedTimes() | unwrap;
        std::println("============================");
        std::println("Staging to src timecost: {} ms", elapsedTime[0]);
        std::println("Dispatch timecost: {} ms", elapsedTime[1]);
        std::println("Dst from staging timecost: {} ms", elapsedTime[2]);
    }

    Timer downloadTimer;
    downloadTimer.begin();
    dstImageBox.download(dstImage.getPData()) | unwrap;
    downloadTimer.end();
    std::println("Download from staging timecost: {} ms", downloadTimer.durationMs());

    dstImage.saveTo("out.png") | unwrap;
}
