#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <vector>

#include "../vkc_bin_helper.hpp"
#include "shader.hpp"
#include "vkc.hpp"

int main() {
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
    srcImageBox.upload(srcImage.getPData()) | unwrap;

    const std::vector descPoolSizes =
        genPoolSizes(srcImageBox, samplerBox, dstImageBox, srcImageBox, samplerBox, dstImageBox);
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

    const std::array grayDLayoutBindings = genDescSetLayoutBindings(srcImageBox, samplerBox, dstImageBox);
    vkc::DescSetLayoutBox grayDLayoutBox = vkc::DescSetLayoutBox::create(pDeviceBox, grayDLayoutBindings) | unwrap;
    const std::array grayDLayoutBoxCRefs{std::cref(grayDLayoutBox)};
    vkc::PipelineLayoutBox grayPLayoutBox = vkc::PipelineLayoutBox::create(pDeviceBox, grayDLayoutBoxCRefs) | unwrap;
    vkc::DescSetsBox grayDescSetsBox = vkc::DescSetsBox::create(pDeviceBox, descPoolBox, grayDLayoutBoxCRefs) | unwrap;
    const std::array grayWriteDescSets = genWriteDescSets(srcImageBox, samplerBox, dstImageBox);
    const std::array grayWriteDescSetss{std::span{grayWriteDescSets.begin(), grayWriteDescSets.end()}};
    grayDescSetsBox.updateDescSets(grayWriteDescSetss);

    // Command Buffer
    vkc::SemaphoreBox semaphoreBox = vkc::SemaphoreBox::create(pDeviceBox) | unwrap;
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox gaussCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;
    vkc::CommandBufferBox grayCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;
    vkc::TimestampQueryPoolBox queryPoolBox =
        vkc::TimestampQueryPoolBox::create(pDeviceBox, 6, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
        unwrap;

    // Pipeline
    constexpr vkc::BlockSize blockSize{16, 16, 1};
    vkc::ShaderBox gaussShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::gaussFilter::v0::code) | unwrap;
    vkc::SpecConstantBox specConstantBox{blockSize.x, blockSize.y};
    vkc::PipelineBox gaussPipelineBox =
        vkc::PipelineBox::createCompute(pDeviceBox, gaussPLayoutBox, gaussShaderBox, specConstantBox.getSpecInfo()) |
        unwrap;

    vkc::ShaderBox grayShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::grayscale::ro::code) | unwrap;
    vkc::PipelineBox grayPipelineBox =
        vkc::PipelineBox::createCompute(pDeviceBox, grayPLayoutBox, grayShaderBox, specConstantBox.getSpecInfo()) |
        unwrap;

    // Gaussian Blur
    gaussCmdBufBox.begin() | unwrap;
    gaussCmdBufBox.bindPipeline(gaussPipelineBox);
    gaussCmdBufBox.bindDescSets(gaussDescSetsBox, gaussPLayoutBox, vk::PipelineBindPoint::eCompute);
    gaussCmdBufBox.pushConstant(kernelSizePcBox, gaussPLayoutBox);
    gaussCmdBufBox.recordResetQueryPool(queryPoolBox);
    gaussCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::SampledImageBox>(srcImageBoxRefs);
    gaussCmdBufBox.recordCopyStagingToSrc(srcImageBox);
    gaussCmdBufBox.recordSrcPrepareShaderRead<vkc::SampledImageBox>(srcImageBoxRefs);
    gaussCmdBufBox.recordDstPrepareShaderWrite(dstImageBoxRefs);
    gaussCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    gaussCmdBufBox.recordDispatch(srcImage.getExtent().extent(), blockSize);
    gaussCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    gaussCmdBufBox.end() | unwrap;
    queueBox.submit(gaussCmdBufBox, semaphoreBox) | unwrap;

    // Grayscale
    grayCmdBufBox.begin() | unwrap;
    grayCmdBufBox.bindPipeline(grayPipelineBox);
    grayCmdBufBox.bindDescSets(grayDescSetsBox, grayPLayoutBox, vk::PipelineBindPoint::eCompute);
    grayCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::SampledImageBox>(srcImageBoxRefs);
    grayCmdBufBox.recordPrepareSendBeforeDispatch(dstImageBoxRefs);
    grayCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
    grayCmdBufBox.recordCopyStorageToAnother(dstImageBox, srcImageBox);
    grayCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
    grayCmdBufBox.recordSrcPrepareShaderRead<vkc::SampledImageBox>(srcImageBoxRefs);
    grayCmdBufBox.recordDstPrepareShaderWrite(dstImageBoxRefs);
    grayCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    grayCmdBufBox.recordDispatch(srcImage.getExtent().extent(), blockSize);
    grayCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    grayCmdBufBox.recordPrepareSendAfterDispatch(dstImageBoxRefs);
    grayCmdBufBox.recordCopyDstToStaging(dstImageBox);
    grayCmdBufBox.recordWaitDownloadComplete(dstImageBoxRefs);
    grayCmdBufBox.end() | unwrap;

    queueBox.submitAndWaitSemaphore(grayCmdBufBox, semaphoreBox, vk::PipelineStageFlagBits::eTransfer, fenceBox) |
        unwrap;
    fenceBox.wait() | unwrap;
    fenceBox.reset() | unwrap;

    auto elapsedTime = queryPoolBox.getElaspedTimes() | unwrap;
    std::println("GaussFilter dispatch timecost: {} ms", elapsedTime[0]);
    std::println("Storage to sampled transfer timecost: {} ms", elapsedTime[1]);
    std::println("Grayscale dispatch timecost: {} ms", elapsedTime[2]);

    dstImageBox.download(dstImage.getPData()) | unwrap;
    dstImage.saveTo("out.png") | unwrap;
}
