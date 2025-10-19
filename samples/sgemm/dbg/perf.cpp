#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <random>
#include <span>
#include <vector>

#include "../../vkc_helper.hpp"
#include "shader.hpp"
#include "vkc.hpp"

int main() {
    vkc::initVulkan() | unwrap;

    constexpr std::array SIZES{1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240};
    constexpr int HEATUP_TIMES = 2;
    constexpr int PERF_TIMES = 5;

    // Device
    vkc::InstanceBox instBox = vkc::InstanceBox::create() | unwrap;
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(instBox) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    vkc::DefaultPhyDeviceFeatures phyDeviceFeatures = vkc::DefaultPhyDeviceFeatures::create(phyDeviceBox) | unwrap;
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::createWithExts(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}, {},
                                       phyDeviceFeatures.getPFeature()) |
        unwrap);
    vkc::QueueBox queueBox = vkc::QueueBox::create(*pDeviceBox, vk::QueueFlagBits::eCompute) | unwrap;

    for (const int size : SIZES) {
        const int M = size;
        const int K = size;
        const int N = size;
        const vkc::Extent extentA{K, M, vk::Format::eR32Sfloat};
        const vkc::Extent extentB{N, K, vk::Format::eR32Sfloat};
        const vkc::Extent extentDst{extentB.width(), extentA.height(), vk::Format::eR32Sfloat};

        // Src data
        std::mt19937 rdEngine;
        rdEngine.seed(37);
        std::uniform_real_distribution dist(0.0f, 1.0f);

        std::vector<float> srcMatA(extentA.elemCount());
        std::vector<float> srcMatB(extentB.elemCount());
        for (auto& val : srcMatA) {
            val = dist(rdEngine);
        }
        for (auto& val : srcMatB) {
            val = dist(rdEngine);
        }

        // Descriptor & Layouts
        vkc::StorageBufferBox srcMatABox =
            vkc::StorageBufferBox::create(pDeviceBox, extentA.size(), vkc::StorageType::ReadOnly) | unwrap;
        vkc::StagingBufferBox srcMatAStagingBufferBox =
            vkc::StagingBufferBox::create(pDeviceBox, extentA.size(), vkc::StorageType::ReadOnly) | unwrap;
        vkc::StorageBufferBox srcMatBBox =
            vkc::StorageBufferBox::create(pDeviceBox, extentB.size(), vkc::StorageType::ReadOnly) | unwrap;
        vkc::StagingBufferBox srcMatBStagingBufferBox =
            vkc::StagingBufferBox::create(pDeviceBox, extentB.size(), vkc::StorageType::ReadOnly) | unwrap;
        const std::array srcMatBoxRefs{std::ref(srcMatABox), std::ref(srcMatBBox)};
        vkc::StorageBufferBox dstMatBox =
            vkc::StorageBufferBox::create(pDeviceBox, extentDst.size(), vkc::StorageType::ReadWrite) | unwrap;
        vkc::StagingBufferBox dstMatStagingBufferBox =
            vkc::StagingBufferBox::create(pDeviceBox, extentDst.size(), vkc::StorageType::ReadWrite) | unwrap;
        const std::array dstMatBoxRefs{std::ref(dstMatBox)};
        const std::array dstStagingBufferRefs{std::ref(dstMatStagingBufferBox)};
        srcMatAStagingBufferBox.upload((std::byte*)srcMatA.data()) | unwrap;
        srcMatBStagingBufferBox.upload((std::byte*)srcMatB.data()) | unwrap;

        const std::vector descPoolSizes = genPoolSizes(srcMatABox, srcMatBBox, dstMatBox);
        vkc::DescPoolBox descPoolBox = vkc::DescPoolBox::create(pDeviceBox, descPoolSizes) | unwrap;

        const std::array sgemmDLayoutBindings = genDescSetLayoutBindings(srcMatABox, srcMatBBox, dstMatBox);
        vkc::DescSetLayoutBox sgemmDLayoutBox =
            vkc::DescSetLayoutBox::create(pDeviceBox, sgemmDLayoutBindings) | unwrap;
        const std::array sgemmDLayoutBoxCRefs{std::cref(sgemmDLayoutBox)};
        vkc::PipelineLayoutBox sgemmPLayoutBox =
            vkc::PipelineLayoutBox::create(pDeviceBox, sgemmDLayoutBoxCRefs) | unwrap;
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
            vkc::TimestampQueryPoolBox::create(pDeviceBox, 2, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
            unwrap;

        // Pipeline
        constexpr int blockTileM = 128;
        constexpr int blockTileN = 64;
        constexpr int blockTileK = 16;
        constexpr int threadTileM = 8;
        constexpr int threadTileN = 8;
        constexpr int threadTileK = 4;
        constexpr int threadSubTileM = 4;
        constexpr int threadSubTileN = 8;
        constexpr int threadSubTileK = 4;
        constexpr int groupSizeX = blockTileN / threadTileN;
        constexpr int groupSizeY = blockTileM / threadTileM;
        const int groupNumX = extentDst.width() / blockTileN;
        const int groupNumY = extentDst.height() / blockTileM;
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::simt::v7::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{
            groupSizeX,     groupSizeY,    M,           N,           K,           blockTileM,
            blockTileN,     blockTileK,    threadTileM, threadTileN, threadTileK, threadSubTileM,
            threadSubTileN, threadSubTileK};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
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

        for (int i = 0; i < HEATUP_TIMES; i++) {
            queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
            fenceBox.wait() | unwrap;
            fenceBox.reset() | unwrap;
        }

        std::vector<float> elapsedTimes;
        for (int i = 0; i < PERF_TIMES; i++) {
            queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
            fenceBox.wait() | unwrap;
            fenceBox.reset() | unwrap;

            auto elapsedTime = queryPoolBox.getElaspedTimes() | unwrap;
            elapsedTimes.push_back(elapsedTime[0]);
        }

        const auto [meanTime, stdTime] = meanStd(elapsedTimes);
        const float macs = (float)M * N * K * 2;
        const float meanTflops = macs / meanTime / 1e9;
        std::println("{}", meanTflops);
    }
}
