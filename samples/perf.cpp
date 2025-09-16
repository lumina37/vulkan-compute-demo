#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <span>

#include "shader.hpp"
#include "vkc.hpp"
#include "vkc_helper.hpp"

int main() {
    vkc::initVulkan() | unwrap;

    constexpr int M = 2048;
    constexpr int K = 2048;
    constexpr int N = 2048;
    constexpr vkc::Extent extentA{K, M, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentB{N, K, vk::Format::eR32Sfloat};
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

    auto& phyDeviceProps = phyDeviceWithProps.getPhyDeviceProps();
    if (!phyDeviceProps.extensions.has(vk::KHRPerformanceQueryExtensionName)) {
        std::println(std::cerr, "VK_KHR_performance_query not supported");
        return -1;
    }
    if (!phyDeviceProps.extensions.has(vk::EXTHostQueryResetExtensionName)) {
        std::println(std::cerr, "VK_EXT_host_query_reset not supported");
        return -1;
    }
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    vkc::DefaultPhyDeviceFeatures phyDeviceFeatures = vkc::DefaultPhyDeviceFeatures::create(phyDeviceBox) | unwrap;

    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    vkc::PerfCounterProps perfProps = vkc::PerfCounterProps::create(phyDeviceBox, computeQFamilyIdx) | unwrap;
    const int perfCounterCount = std::min((int)perfProps.perfCounters.size(), 11);
    for (int i = 0; i < perfCounterCount; i++) {
        const auto& perfCounter = perfProps.perfCounters[i];
        std::println("===============");
        std::println("index: {}", i);
        std::println("name: {}", perfCounter.getName());
        std::println("cate: {}", perfCounter.getCategory());
        std::println("desc: {}", perfCounter.getDescription());
        std::println("unit: {}", vk::to_string(perfCounter.getUnit()));
        std::println("storage: {}", vk::to_string(perfCounter.getStorage()));
    }

    constexpr std::string_view perfQueryExtName{vk::KHRPerformanceQueryExtensionName};
    constexpr std::string_view hostResetExtName{vk::EXTHostQueryResetExtensionName};
    constexpr std::array deviceExtNames{perfQueryExtName, hostResetExtName};
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::createWithExts(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}, deviceExtNames,
                                       phyDeviceFeatures.getPFeature()) |
        unwrap);
    vkc::QueueBox queueBox = vkc::QueueBox::create(*pDeviceBox, vk::QueueFlagBits::eCompute) | unwrap;

    // Descriptor & Layouts
    vkc::StorageImageBox srcMatABox =
        vkc::StorageImageBox::create(pDeviceBox, srcMatA.getExtent(), vkc::StorageImageType::Read) | unwrap;
    vkc::StorageImageBox srcMatBBox =
        vkc::StorageImageBox::create(pDeviceBox, srcMatB.getExtent(), vkc::StorageImageType::Read) | unwrap;
    const std::array srcMatBoxRefs{std::ref(srcMatABox), std::ref(srcMatBBox)};
    vkc::StorageImageBox dstMatBox = vkc::StorageImageBox::create(pDeviceBox, dstMatVk.getExtent()) | unwrap;
    const std::array dstMatBoxRefs{std::ref(dstMatBox)};
    srcMatABox.upload(srcMatA.getPData()) | unwrap;
    srcMatBBox.upload(srcMatB.getPData()) | unwrap;

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
    std::array perfCounterIndices{0u, 8u, 10u};
    vkc::PerfQueryPoolBox perfQueryPoolBox =
        vkc::PerfQueryPoolBox::create(pDeviceBox, computeQFamilyIdx, 1, perfCounterIndices) | unwrap;
    perfQueryPoolBox.hostReset();

    // Pipeline
    constexpr int TM = 4;
    constexpr int TN = TM;
    constexpr int TK = TM;
    constexpr int groupSizeX = 16;
    constexpr int groupSizeY = 16;
    constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), groupSizeX * TN);
    constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), groupSizeY * TM);
    vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::v2::code) | unwrap;
    vkc::SpecConstantBox specConstantBox{groupSizeX, groupSizeY, K, TM, TN, TK};
    vkc::PipelineBox sgemmPipelineBox =
        vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox, specConstantBox.getSpecInfo()) |
        unwrap;

    // Record Command Buffer
    const vkc::ProfilingLockBox lock = vkc::ProfilingLockBox::create(pDeviceBox) | unwrap;
    for (int i = 0; i < 10; i++) {
        perfQueryPoolBox.hostReset();
        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
        sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
        sgemmCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatABox);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatBBox);
        sgemmCmdBufBox.recordSrcPrepareShaderRead<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordDstPrepareShaderWrite(dstMatBoxRefs);
        sgemmCmdBufBox.recordPerfQueryStart(perfQueryPoolBox) | unwrap;
        sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
        sgemmCmdBufBox.recordPerfQueryEnd(perfQueryPoolBox) | unwrap;
        sgemmCmdBufBox.recordPrepareSendAfterDispatch(dstMatBoxRefs);
        sgemmCmdBufBox.recordCopyDstToStaging(dstMatBox);
        sgemmCmdBufBox.recordWaitDownloadComplete(dstMatBoxRefs);
        sgemmCmdBufBox.end() | unwrap;

        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        auto perfQueryResults = perfQueryPoolBox.getResults<uint64_t, uint64_t, float>() | unwrap;
        std::println("============================");
        std::println("GPU Elapsed Time: {} ms", (float)std::get<0>(perfQueryResults[0]) / 1e6);
        std::println("Dispatched Threads: {}", std::get<1>(perfQueryResults[0]));
        std::println("SM Active: {} %", std::get<2>(perfQueryResults[0]));
    }
}
