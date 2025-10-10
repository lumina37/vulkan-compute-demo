#include <expected>
#include <filesystem>
#include <print>
#include <random>
#include <ranges>
#include <span>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "spirv/sgemm.hpp"
#include "vkc.hpp"
#include "vkc_helper.hpp"

namespace fs = std::filesystem;
namespace rgs = std::ranges;

void sgemmRefImpl(const std::span<const float> srcMatA, const std::span<const float> srcMatB,
                  const std::span<float> dstMat, const vkc::Extent extentA, const vkc::Extent extentB) {
    const int M = extentA.height();
    const int N = extentB.width();
    const int K = extentA.width();

    const auto kernelFn = [&](int tx, int ty) {
        float acc = 0;
        for (int k = 0; k < K; k++) {
            acc += srcMatA[ty * K + k] * srcMatB[k * N + tx];
        }
        dstMat[ty * N + tx] = acc;
    };

    for (int dstX = 0; dstX < N; dstX++) {
        for (int dstY = 0; dstY < M; dstY++) {
            kernelFn(dstX, dstY);
        }
    }
}

TEST_CASE("GLSL-SGEMM-TCore", "") {
    vkc::initVulkan() | unwrap;

    constexpr float maxValidDiff = 0.01f;
    constexpr float maxValidAvgDiff = 0.001f;

    constexpr int M = 256;
    constexpr int K = 128;
    constexpr int N = 512;
    constexpr vkc::Extent extentA{K, M, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentB{N, K, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentDst{extentB.width(), extentA.height(), vk::Format::eR32Sfloat};

    // Src data
    vkc::StbImageBox srcMatA = vkc::StbImageBox::createWithExtent(extentA) | unwrap;
    std::span<float> srcSpanA = std::span{(float*)srcMatA.getPData(), extentA.elemCount()};
    vkc::StbImageBox srcMatB = vkc::StbImageBox::createWithExtent(extentB) | unwrap;
    std::span<float> srcSpanB = std::span{(float*)srcMatB.getPData(), extentB.elemCount()};
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (auto& val : srcSpanA) {
        val = dist(rdEngine);
    }
    for (auto& val : srcSpanB) {
        val = dist(rdEngine);
    }

    // CPU Reference
    vkc::StbImageBox dstMatCpuRef = vkc::StbImageBox::createWithExtent(extentDst) | unwrap;
    std::span<float> dstMatCpuRefSpan = std::span{(float*)dstMatCpuRef.getPData(), extentDst.elemCount()};

    sgemmRefImpl(srcSpanA, srcSpanB, dstMatCpuRefSpan, extentA, extentB);

    vkc::StbImageBox dstMatVk = vkc::StbImageBox::createWithExtent(extentDst) | unwrap;

    // Device
    vkc::InstanceBox instBox = vkc::InstanceBox::create() | unwrap;
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(instBox) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
    auto& phyDeviceProps = phyDeviceWithProps.getPhyDeviceProps();
    if (!phyDeviceProps.extensions.has(vk::KHRCooperativeMatrixExtensionName)) {
        std::println(std::cerr, "VK_KHR_cooperative_matrix not supported");
        return;
    }
    if (!phyDeviceProps.extensions.has(vk::KHRVulkanMemoryModelExtensionName)) {
        std::println(std::cerr, "VK_KHR_vulkan_memory_model not supported");
        return;
    }
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    vkc::DefaultPhyDeviceFeatures phyDeviceFeatures = vkc::DefaultPhyDeviceFeatures::create(phyDeviceBox) | unwrap;
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    constexpr std::string_view coopMatExtName{vk::KHRCooperativeMatrixExtensionName};
    constexpr std::string_view memModelExtName{vk::KHRVulkanMemoryModelExtensionName};
    constexpr std::array deviceExtNames{coopMatExtName, memModelExtName};
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

    SECTION("v0") {
        constexpr int MMA_M = 16;
        constexpr int MMA_N = 16;
        constexpr int MMA_K = 16;
        const uint32_t groupSizeX = phyDeviceProps.subgroupSize;
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), MMA_N);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), MMA_M);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::tcore::v0::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, M, N, K, MMA_M, MMA_N, MMA_K};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
        sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
        sgemmCmdBufBox.recordPrepareReceive<vkc::StorageBufferBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatAStagingBufferBox, srcMatABox);
        sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatBStagingBufferBox, srcMatBBox);
        sgemmCmdBufBox.recordPrepareShaderRead<vkc::StorageBufferBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordPrepareShaderWrite(dstMatBoxRefs);
        sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
        sgemmCmdBufBox.recordPrepareSend(dstMatBoxRefs);
        sgemmCmdBufBox.recordCopyBufferToStaging(dstMatBox, dstMatStagingBufferBox);
        sgemmCmdBufBox.recordWaitDownloadComplete(dstStagingBufferRefs);
        sgemmCmdBufBox.end() | unwrap;

        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatStagingBufferBox.download(dstMatVk.getPData()) | unwrap;

        float diffAcc = 0;
        std::span<float> dstMatVkSpan = std::span{(float*)dstMatVk.getPData(), extentDst.elemCount()};
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRefSpan, dstMatVkSpan)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVkSpan.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v0 - average diff = {}", avgDiff);
    }

    SECTION("v1") {
        constexpr int MMA_M = 16;
        constexpr int MMA_N = 16;
        constexpr int MMA_K = 16;
        constexpr int blockTileM = 32;
        constexpr int blockTileN = 32;
        constexpr int blockTileK = 16;
        const uint32_t groupSizeX = phyDeviceProps.subgroupSize * (blockTileM / MMA_M) * (blockTileN / MMA_N);
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), blockTileN);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), blockTileM);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::tcore::v1::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, M,     N,          K,          MMA_M,
                                             MMA_N,      MMA_K, blockTileM, blockTileN, blockTileK};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
        sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
        sgemmCmdBufBox.recordPrepareReceive<vkc::StorageBufferBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatAStagingBufferBox, srcMatABox);
        sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatBStagingBufferBox, srcMatBBox);
        sgemmCmdBufBox.recordPrepareShaderRead<vkc::StorageBufferBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordPrepareShaderWrite(dstMatBoxRefs);
        sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
        sgemmCmdBufBox.recordPrepareSend(dstMatBoxRefs);
        sgemmCmdBufBox.recordCopyBufferToStaging(dstMatBox, dstMatStagingBufferBox);
        sgemmCmdBufBox.recordWaitDownloadComplete(dstStagingBufferRefs);
        sgemmCmdBufBox.end() | unwrap;

        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatStagingBufferBox.download(dstMatVk.getPData()) | unwrap;

        float diffAcc = 0;
        std::span<float> dstMatVkSpan = std::span{(float*)dstMatVk.getPData(), extentDst.elemCount()};
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRefSpan, dstMatVkSpan)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVkSpan.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v1 - average diff = {}", avgDiff);
    }

    SECTION("v2") {
        constexpr int MMA_M = 16;
        constexpr int MMA_N = 16;
        constexpr int MMA_K = 16;
        constexpr int blockTileM = 64;
        constexpr int blockTileN = 64;
        constexpr int blockTileK = 16;
        constexpr int wrapTileM = 64;
        constexpr int wrapTileN = 64;
        constexpr int wrapTileK = 16;
        const uint32_t groupSizeX = phyDeviceProps.subgroupSize * (blockTileM / wrapTileM) * (blockTileN / wrapTileN);
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), blockTileN);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), blockTileM);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::tcore::v2::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, M,          N,          K,         MMA_M,     MMA_N,    MMA_K,
                                             blockTileM, blockTileN, blockTileK, wrapTileM, wrapTileN, wrapTileK};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
        sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
        sgemmCmdBufBox.recordPrepareReceive<vkc::StorageBufferBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatAStagingBufferBox, srcMatABox);
        sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatBStagingBufferBox, srcMatBBox);
        sgemmCmdBufBox.recordPrepareShaderRead<vkc::StorageBufferBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordPrepareShaderWrite(dstMatBoxRefs);
        sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
        sgemmCmdBufBox.recordPrepareSend(dstMatBoxRefs);
        sgemmCmdBufBox.recordCopyBufferToStaging(dstMatBox, dstMatStagingBufferBox);
        sgemmCmdBufBox.recordWaitDownloadComplete(dstStagingBufferRefs);
        sgemmCmdBufBox.end() | unwrap;

        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatStagingBufferBox.download(dstMatVk.getPData()) | unwrap;

        float diffAcc = 0;
        std::span<float> dstMatVkSpan = std::span{(float*)dstMatVk.getPData(), extentDst.elemCount()};
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRefSpan, dstMatVkSpan)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVkSpan.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v2 - average diff = {}", avgDiff);
    }

    SECTION("v3") {
        constexpr int MMA_M = 16;
        constexpr int MMA_N = 16;
        constexpr int MMA_K = 16;
        constexpr int blockTileM = 64;
        constexpr int blockTileN = 64;
        constexpr int blockTileK = 16;
        constexpr int wrapTileM = 64;
        constexpr int wrapTileN = 64;
        constexpr int wrapTileK = 16;
        constexpr int stages = 2;
        const uint32_t groupSizeX = phyDeviceProps.subgroupSize * (blockTileM / wrapTileM) * (blockTileN / wrapTileN);
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), blockTileN);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), blockTileM);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::tcore::v3::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, M,         N,          K,          MMA_M,
                                             MMA_N,      MMA_K,     blockTileM, blockTileN, blockTileK,
                                             wrapTileM,  wrapTileN, wrapTileK,  stages};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
        sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
        sgemmCmdBufBox.recordPrepareReceive<vkc::StorageBufferBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatAStagingBufferBox, srcMatABox);
        sgemmCmdBufBox.recordCopyStagingToBuffer(srcMatBStagingBufferBox, srcMatBBox);
        sgemmCmdBufBox.recordPrepareShaderRead<vkc::StorageBufferBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordPrepareShaderWrite(dstMatBoxRefs);
        sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
        sgemmCmdBufBox.recordPrepareSend(dstMatBoxRefs);
        sgemmCmdBufBox.recordCopyBufferToStaging(dstMatBox, dstMatStagingBufferBox);
        sgemmCmdBufBox.recordWaitDownloadComplete(dstStagingBufferRefs);
        sgemmCmdBufBox.end() | unwrap;

        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatStagingBufferBox.download(dstMatVk.getPData()) | unwrap;

        float diffAcc = 0;
        std::span<float> dstMatVkSpan = std::span{(float*)dstMatVk.getPData(), extentDst.elemCount()};
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRefSpan, dstMatVkSpan)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVkSpan.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v3 - average diff = {}", avgDiff);
    }
}
