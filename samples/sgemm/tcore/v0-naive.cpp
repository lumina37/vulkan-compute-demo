#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <vector>

#include "../../vkc_helper.hpp"
#include "shader.hpp"
#include "vkc.hpp"

int main() {
    vkc::initVulkan() | unwrap;

    constexpr int M = 2048;
    constexpr int K = 2048;
    constexpr int N = 2048;
    constexpr vkc::Extent extentA{K, M, vk::Format::eR16Sfloat};
    constexpr vkc::Extent extentB{N, K, vk::Format::eR16Sfloat};
    constexpr vkc::Extent extentDst{extentB.width(), extentA.height(), vk::Format::eR32Sfloat};

    // Src data
    vkc::StbImageBox srcMatA = vkc::StbImageBox::createWithExtent(extentA) | unwrap;
    vkc::StbImageBox srcMatB = vkc::StbImageBox::createWithExtent(extentB) | unwrap;
    vkc::StbImageBox dstMatVk = vkc::StbImageBox::createWithExtent(extentDst) | unwrap;

    // Device
    vkc::DefaultInstanceProps instProps = vkc::DefaultInstanceProps::create() | unwrap;
    if (!instProps.layers.has("VK_LAYER_KHRONOS_validation")) {
        std::println(std::cerr, "VK_LAYER_KHRONOS_validation not supported");
        return -1;
    }
    vkc::InstanceBox instBox = vkc::InstanceBox::create() | unwrap;
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(instBox) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();

    constexpr std::string_view coopMatExtName{vk::KHRCooperativeMatrixExtensionName};
    constexpr std::string_view memModelExtName{vk::KHRVulkanMemoryModelExtensionName};
    constexpr std::array deviceExtNames{coopMatExtName, memModelExtName};
    auto& phyDeviceProps = phyDeviceWithProps.getPhyDeviceProps();
    for (const auto& deviceExtName : deviceExtNames) {
        if (!phyDeviceProps.extensions.has(deviceExtName)) {
            std::println(std::cerr, "{} not supported", deviceExtName);
            return -1;
        }
    }
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    vkc::DefaultPhyDeviceFeatures phyDeviceFeatures = vkc::DefaultPhyDeviceFeatures::create(phyDeviceBox) | unwrap;

    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::createWithExts(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}, deviceExtNames,
                                       phyDeviceFeatures.getPFeature()) |
        unwrap);
    vkc::QueueBox queueBox = vkc::QueueBox::create(*pDeviceBox, vk::QueueFlagBits::eCompute) | unwrap;

    // Descriptor & Layouts
    vkc::StorageBufferBox srcMatABox =
        vkc::StorageBufferBox::create(pDeviceBox, srcMatA.getExtent().size(), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StagingBufferBox srcMatAStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, srcMatABox.getSize(), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StorageBufferBox srcMatBBox =
        vkc::StorageBufferBox::create(pDeviceBox, srcMatB.getExtent().size(), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StagingBufferBox srcMatBStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, srcMatBBox.getSize(), vkc::StorageType::ReadOnly) | unwrap;
    const std::array srcMatBoxRefs{std::ref(srcMatABox), std::ref(srcMatBBox)};
    vkc::StorageBufferBox dstMatBox =
        vkc::StorageBufferBox::create(pDeviceBox, dstMatVk.getExtent().size(), vkc::StorageType::ReadWrite) | unwrap;
    vkc::StagingBufferBox dstMatStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, dstMatVk.getExtent().size(), vkc::StorageType::ReadWrite) | unwrap;
    const std::array dstMatBoxRefs{std::ref(dstMatBox)};
    const std::array dstStagingBufferRefs{std::ref(dstMatStagingBufferBox)};
    srcMatAStagingBufferBox.upload(srcMatA.getPData()) | unwrap;
    srcMatBStagingBufferBox.upload(srcMatB.getPData()) | unwrap;

    const std::vector descPoolSizes = genPoolSizes(srcMatABox, srcMatBBox, dstMatBox);
    vkc::DescPoolBox descPoolBox = vkc::DescPoolBox::create(pDeviceBox, descPoolSizes) | unwrap;

    const std::array sgemmDLayoutBindings = genDescSetLayoutBindings(srcMatABox, srcMatBBox, dstMatBox);
    vkc::DescSetLayoutBox sgemmDLayoutBox = vkc::DescSetLayoutBox::create(pDeviceBox, sgemmDLayoutBindings) | unwrap;
    const std::array sgemmDLayoutBoxCRefs{std::cref(sgemmDLayoutBox)};
    vkc::PipelineLayoutBox sgemmPLayoutBox = vkc::PipelineLayoutBox::create(pDeviceBox, sgemmDLayoutBoxCRefs) | unwrap;
    vkc::DescSetsBox sgemmDescSetsBox =
        vkc::DescSetsBox::create(pDeviceBox, descPoolBox, sgemmDLayoutBoxCRefs) | unwrap;
    const std::array sgemmWriteDescSets = genWriteDescSets(srcMatABox, srcMatBBox, dstMatBox);
    const std::array sgemmWriteDescSetss{std::span{sgemmWriteDescSets.begin(), sgemmWriteDescSets.end()}};
    sgemmDescSetsBox.updateDescSets(sgemmWriteDescSetss);

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox sgemmCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;
    vkc::TimestampQueryPoolBox queryPoolBox =
        vkc::TimestampQueryPoolBox::create(pDeviceBox, 6, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
        unwrap;

    // Pipeline
    constexpr int MMA_N = 16;
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 16;
    const uint32_t groupSizeX = phyDeviceProps.subgroupSize;
    constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), MMA_N);
    constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), MMA_M);
    vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::tcore::v0::code) | unwrap;
    vkc::SpecConstantBox specConstantBox{groupSizeX, M, N, K, MMA_M, MMA_N, MMA_K};
    vkc::PipelineBox sgemmPipelineBox =
        vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox, specConstantBox.getSpecInfo()) |
        unwrap;

    // Record Command Buffer
    sgemmCmdBufBox.begin() | unwrap;
    sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
    sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
    sgemmCmdBufBox.recordResetQueryPool(queryPoolBox);
    sgemmCmdBufBox.recordPrepareReceive<vkc::StorageBufferBox>(srcMatBoxRefs);
    sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatAStagingBufferBox, srcMatABox);
    sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatBStagingBufferBox, srcMatBBox);
    sgemmCmdBufBox.recordPrepareShaderRead<vkc::StorageBufferBox>(srcMatBoxRefs);
    sgemmCmdBufBox.recordPrepareShaderWrite(dstMatBoxRefs);
    sgemmCmdBufBox.recordTimestampStart(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
    sgemmCmdBufBox.recordTimestampEnd(queryPoolBox, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    sgemmCmdBufBox.recordPrepareSend(dstMatBoxRefs);
    sgemmCmdBufBox.recordCopyBufferToStaging(dstMatBox, dstMatStagingBufferBox);
    sgemmCmdBufBox.recordWaitDownloadComplete(dstStagingBufferRefs);
    sgemmCmdBufBox.end() | unwrap;

    for (int i = 0; i < 15; i++) {
        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        auto elapsedTime = queryPoolBox.getElaspedTimes() | unwrap;
        std::println("============================");
        std::println("Dispatch timecost: {} ms", elapsedTime[0]);
    }
}
