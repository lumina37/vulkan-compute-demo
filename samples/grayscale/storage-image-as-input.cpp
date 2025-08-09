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
    vkc::StorageImageBox srcImageBox =
        vkc::StorageImageBox::create(phyDeviceBox, pDeviceBox, srcImage.getExtent(), vkc::StorageImageType::Read) |
        unwrap;
    const std::array srcImageBoxRefs{std::ref(srcImageBox)};
    vkc::StorageImageBox dstImageBox =
        vkc::StorageImageBox::create(phyDeviceBox, pDeviceBox, srcImage.getExtent()) | unwrap;
    const std::array dstImageBoxRefs{std::ref(dstImageBox)};

    Timer uploadTimer;
    uploadTimer.begin();
    srcImageBox.upload(srcImage.getPData()) | unwrap;
    uploadTimer.end();
    std::println("Upload to staging timecost: {} ms", uploadTimer.durationMs());

    const std::vector descPoolSizes = genPoolSizes(srcImageBox, dstImageBox);
    vkc::DescPoolBox descPoolBox = vkc::DescPoolBox::create(pDeviceBox, descPoolSizes) | unwrap;

    const std::array grayDLayoutBindings = genDescSetLayoutBindings(srcImageBox, dstImageBox);
    vkc::DescSetLayoutBox grayDLayoutBox = vkc::DescSetLayoutBox::create(pDeviceBox, grayDLayoutBindings) | unwrap;
    const std::array grayDLayoutBoxCRefs{std::cref(grayDLayoutBox)};
    vkc::PipelineLayoutBox grayPLayoutBox = vkc::PipelineLayoutBox::create(pDeviceBox, grayDLayoutBoxCRefs) | unwrap;
    vkc::DescSetsBox grayDescSetsBox = vkc::DescSetsBox::create(pDeviceBox, descPoolBox, grayDLayoutBoxCRefs) | unwrap;
    const std::array grayWriteDescSets = genWriteDescSets(srcImageBox, dstImageBox);
    const std::array grayWriteDescSetss{std::span{grayWriteDescSets.begin(), grayWriteDescSets.end()}};
    grayDescSetsBox.updateDescSets(grayWriteDescSetss);

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox grayCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;
    vkc::TimestampQueryPoolBox queryPoolBox =
        vkc::TimestampQueryPoolBox::create(pDeviceBox, 6, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
        unwrap;

    // Pipeline
    constexpr int groupSizeX = 16;
    constexpr int groupSizeY = 16;
    const int groupNumX = vkc::ceilDiv(dstImage.getExtent().width(), groupSizeX);
    const int groupNumY = vkc::ceilDiv(dstImage.getExtent().height(), groupSizeY);
    vkc::ShaderBox grayShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::grayscale::rw::code) | unwrap;
    vkc::SpecConstantBox specConstantBox{groupSizeX, groupSizeY};
    vkc::PipelineBox grayPipelineBox =
        vkc::PipelineBox::createCompute(pDeviceBox, grayPLayoutBox, grayShaderBox, specConstantBox.getSpecInfo()) |
        unwrap;

    // Gaussian Blur
    for (int i = 0; i < 15; i++) {
        grayCmdBufBox.begin() | unwrap;
        grayCmdBufBox.bindPipeline(grayPipelineBox);
        grayCmdBufBox.bindDescSets(grayDescSetsBox, grayPLayoutBox, vk::PipelineBindPoint::eCompute);
        grayCmdBufBox.recordResetQueryPool(queryPoolBox);
        grayCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::StorageImageBox>(srcImageBoxRefs);
        grayCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        grayCmdBufBox.recordCopyStagingToSrc(srcImageBox);
        grayCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        grayCmdBufBox.recordSrcPrepareShaderRead<vkc::StorageImageBox>(srcImageBoxRefs);
        grayCmdBufBox.recordDstPrepareShaderWrite(dstImageBoxRefs);
        grayCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        grayCmdBufBox.recordDispatch(groupNumX, groupNumY);
        grayCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        grayCmdBufBox.recordPrepareSendAfterDispatch(dstImageBoxRefs);
        grayCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        grayCmdBufBox.recordCopyDstToStaging(dstImageBox);
        grayCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        grayCmdBufBox.recordWaitDownloadComplete(dstImageBoxRefs);
        grayCmdBufBox.end() | unwrap;

        queueBox.submit(grayCmdBufBox, fenceBox) | unwrap;
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
