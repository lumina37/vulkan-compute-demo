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

void sgemmRefImplWithRowMajorB(const std::span<const float> srcMatA, const std::span<const float> srcMatB,
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

    for (int dstY = 0; dstY < M; dstY++) {
        for (int dstX = 0; dstX < N; dstX++) {
            kernelFn(dstX, dstY);
        }
    }
}

void sgemmRefImplWithColMajorB(const std::span<const float> srcMatQ, const std::span<const float> srcMatK,
                               const std::span<float> dstMat, const vkc::Extent extentA, const vkc::Extent extentB) {
    const int M = extentA.height();
    const int N = extentB.height();
    const int K = extentA.width();

    const auto kernelFn = [&](int tx, int ty) {
        float acc = 0;
        for (int k = 0; k < K; k++) {
            acc += srcMatQ[ty * K + k] * srcMatK[tx * K + k];
        }
        dstMat[tx * N + ty] = acc;
    };

    for (int dstY = 0; dstY < M; dstY++) {
        for (int dstX = 0; dstX < N; dstX++) {
            kernelFn(dstX, dstY);
        }
    }
}

TEST_CASE("GLSL-SGEMM-DBG", "") {
    vkc::initVulkan() | unwrap;

    constexpr float maxValidDiff = 0.001f;
    constexpr float maxValidAvgDiff = 0.0001f;

    constexpr int M = 256;
    constexpr int K = 128;
    constexpr int N = 512;
    constexpr vkc::Extent extentA{K, M, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentB{N, K, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentDst{extentB.width(), extentA.height(), vk::Format::eR32Sfloat};

    // Src data
    std::vector<float> srcMatA(extentA.elemCount());
    std::vector<float> srcMatB(extentB.elemCount());
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (auto& val : srcMatA) {
        val = dist(rdEngine);
    }
    for (auto& val : srcMatB) {
        val = dist(rdEngine);
    }

    // CPU Reference
    std::vector<float> dstMatCpuRef(extentDst.elemCount());
    sgemmRefImplWithRowMajorB(srcMatA, srcMatB, dstMatCpuRef, extentA, extentB);

    std::vector<float> dstMatVk(extentDst.elemCount());

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

    SECTION("wt0") {
        constexpr int blockTileM = 128;
        constexpr int blockTileN = 128;
        constexpr int blockTileK = 16;
        constexpr int wrapTileM = 64;
        constexpr int wrapTileN = 32;
        constexpr int threadTileM = 8;
        constexpr int threadTileN = 4;
        constexpr int wrapMIter = 2;
        constexpr int wrapNIter = 1;
        constexpr int wrapCountY = blockTileM / wrapTileM;
        constexpr int wrapCountX = blockTileN / wrapTileN;
        const int wrapSize = phyDeviceWithProps.getPhyDeviceProps().subgroupSize;
        const int groupSize = wrapSize * (wrapCountX * wrapCountY);
        const int wrapElemCount = wrapSize * (threadTileM * threadTileN * wrapMIter * wrapNIter);
        constexpr int expectWrapElemCount = wrapTileM * wrapTileN;
        if (wrapElemCount != expectWrapElemCount) {
            throw std::format("launch param error, {} != {}", wrapElemCount, expectWrapElemCount);
        }
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), blockTileN);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), blockTileM);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::dbg::wt0::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSize,  M,           N,          K,         blockTileM,
                                             blockTileN, blockTileK,  wrapTileM,  wrapTileN, wrapMIter,
                                             wrapNIter,  threadTileM, threadTileN};
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
        std::println("wt0 - average diff = {}", avgDiff);
    }
}

TEST_CASE("GLSL-SGEMM-GGML", "") {
    vkc::initVulkan() | unwrap;

    constexpr float maxValidDiff = 0.001f;
    constexpr float maxValidAvgDiff = 0.0001f;

    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 256;
    constexpr vkc::Extent extentA{K, M, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentB{K, N, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentDst{extentB.height(), extentA.height(), vk::Format::eR32Sfloat};

    // Src data
    std::vector<float> srcMatA(extentA.elemCount());
    std::vector<float> srcMatB(extentB.elemCount());
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (auto& val : srcMatA) {
        val = dist(rdEngine);
    }
    for (auto& val : srcMatB) {
        val = dist(rdEngine);
    }

    // CPU Reference
    std::vector<float> dstMatCpuRef(extentDst.elemCount());
    sgemmRefImplWithColMajorB(srcMatA, srcMatB, dstMatCpuRef, extentA, extentB);

    std::vector<float> dstMatVk(extentDst.elemCount());

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

    struct GGMLPushConstant {
        int M;
        int N;
        int K;
        int stride_a;
        int stride_b;
        int stride_d;

        int batch_stride_a;
        int batch_stride_b;
        int batch_stride_d;

        int k_split;
        int ne02;
        int ne12;
        int broadcast2;
        int broadcast3;
    };

    GGMLPushConstant sgemmPushConstant{M, N, K, K, K, M, M * K, K * N, M * N, K, 1, 1, 1, 1};
    vkc::PushConstantBox sgemmPushConstantBox = vkc::PushConstantBox{sgemmPushConstant};
    vkc::PipelineLayoutBox sgemmPLayoutBox =
        vkc::PipelineLayoutBox::createWithPushConstant(pDeviceBox, sgemmDLayoutBoxCRefs,
                                                       sgemmPushConstantBox.getPushConstantRange()) |
        unwrap;
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

    SECTION("ggml") {
        constexpr int groupSize = 128;
        constexpr int blockTileM = 128;
        constexpr int blockTileN = 128;
        constexpr int blockTileK = 16;
        constexpr int warpTileM = 64;
        constexpr int warpTileN = 64;
        constexpr int warpTileMIter = 2;
        constexpr int threadTileM = 4;
        constexpr int threadTileN = 4;
        constexpr int threadTileK = 1;
        constexpr int warpSize = 32;
        const int groupNumX = vkc::ceilDiv(extentDst.width(), blockTileM);
        const int groupNumY = vkc::ceilDiv(extentDst.height(), blockTileN);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::dbg::ggml::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSize,     blockTileM,  blockTileN,  blockTileK,  warpTileM, warpTileN,
                                             warpTileMIter, threadTileM, threadTileN, threadTileK, warpSize};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.pushConstant(sgemmPushConstantBox, sgemmPLayoutBox);
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
        int count = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRef, dstMatVk)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
            count++;
        }
        float avgDiff = diffAcc / (float)dstMatVk.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("ggml - average diff = {}", avgDiff);
    }
}
