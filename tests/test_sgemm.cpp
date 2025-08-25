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

TEST_CASE("CPU-SGEMM", "") {
    constexpr vkc::Extent extentA{3, 2, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentB{1, 3, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentDst{extentB.width(), extentA.height(), vk::Format::eR32Sfloat};

    // Src data
    vkc::StbImageBox srcMatA = vkc::StbImageBox::createWithExtent(extentA) | unwrap;
    std::span<float> srcSpanA = std::span{(float*)srcMatA.getPData(), extentA.elemCount()};
    // srcMatA = [[1,2,3],[4,5,6]]
    for (int i = 0; i < srcSpanA.size(); i++) {
        srcSpanA[i] = (float)i + 1;
    }

    vkc::StbImageBox srcMatB = vkc::StbImageBox::createWithExtent(extentB) | unwrap;
    std::span<float> srcSpanB = std::span{(float*)srcMatB.getPData(), extentB.elemCount()};
    // srcMatB = [[1,2,3]]
    for (int i = 0; i < srcSpanB.size(); i++) {
        srcSpanB[i] = (float)i + 1;
    }

    // CPU Reference
    vkc::StbImageBox dstMatCpuRef = vkc::StbImageBox::createWithExtent(srcMatA.getExtent()) | unwrap;
    std::span<float> dstSpan = std::span{(float*)dstMatCpuRef.getPData(), extentDst.elemCount()};

    sgemmRefImpl(srcSpanA, srcSpanB, dstSpan, extentA, extentB);
    REQUIRE(dstSpan[0] == Catch::Approx(14.f));
    REQUIRE(dstSpan[1] == Catch::Approx(32.f));
}

TEST_CASE("GLSL-SGEMM", "") {
    vkc::initVulkan() | unwrap;

    constexpr float maxValidDiff = 0.0001f;
    constexpr float maxValidAvgDiff = 0.000001f;

    constexpr int M = 128;
    constexpr int K = 64;
    constexpr int N = 256;
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
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::create(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}) | unwrap);
    vkc::QueueBox queueBox = vkc::QueueBox::create(*pDeviceBox, vk::QueueFlagBits::eCompute) | unwrap;

    // Descriptor & Layouts
    vkc::StorageImageBox srcMatABox =
        vkc::StorageImageBox::create(phyDeviceBox, pDeviceBox, srcMatA.getExtent(), vkc::StorageImageType::Read) |
        unwrap;
    vkc::StorageImageBox srcMatBBox =
        vkc::StorageImageBox::create(phyDeviceBox, pDeviceBox, srcMatB.getExtent(), vkc::StorageImageType::Read) |
        unwrap;
    const std::array srcMatBoxRefs{std::ref(srcMatABox), std::ref(srcMatBBox)};
    vkc::StorageImageBox dstMatBox =
        vkc::StorageImageBox::create(phyDeviceBox, pDeviceBox, dstMatVk.getExtent()) | unwrap;
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

    SECTION("v0") {
        constexpr int groupSizeX = 16;
        constexpr int groupSizeY = 16;
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), groupSizeX);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), groupSizeY);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::v0::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, groupSizeY};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
        sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
        sgemmCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatABox);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatBBox);
        sgemmCmdBufBox.recordSrcPrepareShaderRead<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordDstPrepareShaderWrite(dstMatBoxRefs);
        sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
        sgemmCmdBufBox.recordPrepareSendAfterDispatch(dstMatBoxRefs);
        sgemmCmdBufBox.recordCopyDstToStaging(dstMatBox);
        sgemmCmdBufBox.recordWaitDownloadComplete(dstMatBoxRefs);
        sgemmCmdBufBox.end() | unwrap;

        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatBox.download(dstMatVk.getPData()) | unwrap;

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
        constexpr int groupSizeX = 16;
        constexpr int groupSizeY = 16;
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), groupSizeX);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), groupSizeY);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::v1::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, K};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
        sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
        sgemmCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatABox);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatBBox);
        sgemmCmdBufBox.recordSrcPrepareShaderRead<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordDstPrepareShaderWrite(dstMatBoxRefs);
        sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
        sgemmCmdBufBox.recordPrepareSendAfterDispatch(dstMatBoxRefs);
        sgemmCmdBufBox.recordCopyDstToStaging(dstMatBox);
        sgemmCmdBufBox.recordWaitDownloadComplete(dstMatBoxRefs);
        sgemmCmdBufBox.end() | unwrap;

        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatBox.download(dstMatVk.getPData()) | unwrap;

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
        constexpr int TM = 4;
        constexpr int TN = TM;
        constexpr int TK = TM;
        constexpr int groupSizeX = 16;
        constexpr int groupSizeY = 16;
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), groupSizeX * TN);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), groupSizeY * TM);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::v2::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, groupSizeY, K, TM, TN, TK};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
        sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
        sgemmCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatABox);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatBBox);
        sgemmCmdBufBox.recordSrcPrepareShaderRead<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordDstPrepareShaderWrite(dstMatBoxRefs);
        sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
        sgemmCmdBufBox.recordPrepareSendAfterDispatch(dstMatBoxRefs);
        sgemmCmdBufBox.recordCopyDstToStaging(dstMatBox);
        sgemmCmdBufBox.recordWaitDownloadComplete(dstMatBoxRefs);
        sgemmCmdBufBox.end() | unwrap;

        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatBox.download(dstMatVk.getPData()) | unwrap;

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
        constexpr int groupSizeX = 16;
        constexpr int groupSizeY = 16;
        constexpr int groupNumX = vkc::ceilDiv(extentDst.width(), groupSizeX);
        constexpr int groupNumY = vkc::ceilDiv(extentDst.height(), groupSizeY);
        vkc::ShaderBox sgemmShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::v3::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{K};
        vkc::PipelineBox sgemmPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, sgemmPLayoutBox, sgemmShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        sgemmCmdBufBox.begin() | unwrap;
        sgemmCmdBufBox.bindPipeline(sgemmPipelineBox);
        sgemmCmdBufBox.bindDescSets(sgemmDescSetsBox, sgemmPLayoutBox, vk::PipelineBindPoint::eCompute);
        sgemmCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatABox);
        sgemmCmdBufBox.recordCopyStagingToSrc(srcMatBBox);
        sgemmCmdBufBox.recordSrcPrepareShaderRead<vkc::StorageImageBox>(srcMatBoxRefs);
        sgemmCmdBufBox.recordDstPrepareShaderWrite(dstMatBoxRefs);
        sgemmCmdBufBox.recordDispatch(groupNumX, groupNumY);
        sgemmCmdBufBox.recordPrepareSendAfterDispatch(dstMatBoxRefs);
        sgemmCmdBufBox.recordCopyDstToStaging(dstMatBox);
        sgemmCmdBufBox.recordWaitDownloadComplete(dstMatBoxRefs);
        sgemmCmdBufBox.end() | unwrap;

        queueBox.submit(sgemmCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatBox.download(dstMatVk.getPData()) | unwrap;

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
}
