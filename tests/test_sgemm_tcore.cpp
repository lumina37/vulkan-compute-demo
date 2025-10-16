#include <expected>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "spirv/sgemm.hpp"
#include "vkc.hpp"
#include "vkc_helper.hpp"

namespace rgs = std::ranges;

void sgemmRefImpl(const std::span<const float16> srcMatA, const std::span<const float16> srcMatB,
                  const std::span<float> dstMat, const vkc::Extent extentA, const vkc::Extent extentB) {
    const int M = extentA.height();
    const int N = extentB.width();
    const int K = extentA.width();

    const auto kernelFn = [&](int tx, int ty) {
        float acc = 0;
        for (int k = 0; k < K; k++) {
            acc += (float)srcMatA[ty * K + k] * (float)srcMatB[k * N + tx];
        }
        dstMat[ty * N + tx] = acc;
    };

    for (int dstY = 0; dstY < M; dstY++) {
        for (int dstX = 0; dstX < N; dstX++) {
            kernelFn(dstX, dstY);
        }
    }
}

TEST_CASE("GLSL-SGEMM-TCore", "") {
    vkc::initVulkan() | unwrap;

    constexpr float maxValidDiff = 0.001f;
    constexpr float maxValidAvgDiff = 0.0001f;

    constexpr int M = 256;
    constexpr int K = 128;
    constexpr int N = 512;
    constexpr vkc::Extent extentA{K, M, vk::Format::eR16Sfloat};
    constexpr vkc::Extent extentB{N, K, vk::Format::eR16Sfloat};
    constexpr vkc::Extent extentDst{extentB.width(), extentA.height(), vk::Format::eR32Sfloat};

    // Src data
    std::vector<float16> srcMatA(extentA.elemCount());
    std::vector<float16> srcMatB(extentB.elemCount());
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (auto& val : srcMatA) {
        val = (float16)dist(rdEngine);
    }
    for (auto& val : srcMatB) {
        val = (float16)dist(rdEngine);
    }

    // CPU Reference
    std::vector<float> dstMatCpuRef(extentDst.elemCount());
    sgemmRefImpl(srcMatA, srcMatB, dstMatCpuRef, extentA, extentB);

    std::vector<float> dstMatVk(extentDst.elemCount());

    // Device
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
            return;
        }
    }
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();

    using PhyDeviceFeatures =
        vkc::PhyDeviceFeatures_<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                                vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>;
    PhyDeviceFeatures phyDeviceFeatures = PhyDeviceFeatures::create(phyDeviceBox) | unwrap;

    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::createWithExts(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}, deviceExtNames,
                                       phyDeviceFeatures.getPFeature()) |
        unwrap);
    vkc::QueueBox queueBox = vkc::QueueBox::create(*pDeviceBox, vk::QueueFlagBits::eCompute) | unwrap;

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

        dstMatStagingBufferBox.download((std::byte*)dstMatVk.data()) | unwrap;

        float diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRef, dstMatVk)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVk.size();

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

        dstMatStagingBufferBox.download((std::byte*)dstMatVk.data()) | unwrap;

        float diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRef, dstMatVk)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVk.size();

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

        dstMatStagingBufferBox.download((std::byte*)dstMatVk.data()) | unwrap;

        float diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRef, dstMatVk)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVk.size();

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
        const uint32_t groupSizeX = phyDeviceProps.subgroupSize * (blockTileM / wrapTileM) * (blockTileN / wrapTileN);
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), blockTileN);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), blockTileM);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::tcore::v3::code) | unwrap;
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

        dstMatStagingBufferBox.download((std::byte*)dstMatVk.data()) | unwrap;

        float diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRef, dstMatVk)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVk.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v3 - average diff = {}", avgDiff);
    }

    SECTION("v4") {
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
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::tcore::v4::code) | unwrap;
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

        dstMatStagingBufferBox.download((std::byte*)dstMatVk.data()) | unwrap;

        float diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRef, dstMatVk)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVk.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v4 - average diff = {}", avgDiff);
    }
}

TEST_CASE("GLSL-SGEMM-TCore-NV", "") {
    vkc::initVulkan() | unwrap;

    constexpr float maxValidDiff = 0.001f;
    constexpr float maxValidAvgDiff = 0.0001f;

    constexpr int M = 256;
    constexpr int K = 128;
    constexpr int N = 512;
    constexpr vkc::Extent extentA{K, M, vk::Format::eR16Sfloat};
    constexpr vkc::Extent extentB{N, K, vk::Format::eR16Sfloat};
    constexpr vkc::Extent extentDst{extentB.width(), extentA.height(), vk::Format::eR32Sfloat};

    // Src data
    std::vector<float16> srcMatA(extentA.elemCount());
    std::vector<float16> srcMatB(extentB.elemCount());
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (auto& val : srcMatA) {
        val = (float16)dist(rdEngine);
    }
    for (auto& val : srcMatB) {
        val = (float16)dist(rdEngine);
    }

    // CPU Reference
    std::vector<float> dstMatCpuRef(extentDst.elemCount());
    sgemmRefImpl(srcMatA, srcMatB, dstMatCpuRef, extentA, extentB);

    std::vector<float> dstMatVk(extentDst.elemCount());

    // Device
    vkc::InstanceBox instBox = vkc::InstanceBox::create() | unwrap;
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(instBox) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
    constexpr std::string_view coopMatExtName{vk::KHRCooperativeMatrixExtensionName};
    constexpr std::string_view coopMat2ExtName{vk::NVCooperativeMatrix2ExtensionName};
    constexpr std::string_view memModelExtName{vk::KHRVulkanMemoryModelExtensionName};
    constexpr std::array deviceExtNames{coopMatExtName, coopMat2ExtName, memModelExtName};
    auto& phyDeviceProps = phyDeviceWithProps.getPhyDeviceProps();
    for (const auto& deviceExtName : deviceExtNames) {
        if (!phyDeviceProps.extensions.has(deviceExtName)) {
            std::println(std::cerr, "{} not supported", deviceExtName);
            return;
        }
    }
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    using PhyDeviceFeatures =
        vkc::PhyDeviceFeatures_<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                                vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR,
                                vk::PhysicalDeviceCooperativeMatrix2FeaturesNV>;
    PhyDeviceFeatures phyDeviceFeatures = PhyDeviceFeatures::create(phyDeviceBox) | unwrap;
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::createWithExts(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}, deviceExtNames,
                                       phyDeviceFeatures.getPFeature()) |
        unwrap);
    vkc::QueueBox queueBox = vkc::QueueBox::create(*pDeviceBox, vk::QueueFlagBits::eCompute) | unwrap;

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

    SECTION("v5") {
        constexpr int blockTileM = 64;
        constexpr int blockTileN = 64;
        constexpr int blockTileK = 16;
        constexpr int groupSize = 128;
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), blockTileN);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), blockTileM);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::tcore::v5::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSize, M, N, K, blockTileM, blockTileN, blockTileK};
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

        dstMatStagingBufferBox.download((std::byte*)dstMatVk.data()) | unwrap;

        float diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRef, dstMatVk)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVk.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v5 - average diff = {}", avgDiff);
    }
}
