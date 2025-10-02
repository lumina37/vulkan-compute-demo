#include <expected>
#include <filesystem>
#include <print>
#include <random>
#include <ranges>
#include <span>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "spirv/flash_attention2.hpp"
#include "vkc.hpp"
#include "vkc_helper.hpp"

namespace fs = std::filesystem;
namespace rgs = std::ranges;

void fa2RefImpl(const std::span<const float> srcMatQ, const std::span<const float> srcMatK,
                const std::span<const float> srcMatV, const std::span<float> dstMatO, const vkc::Extent extent) {
    const int N = extent.height();
    const int d = extent.width();
    const float invSqrtD = std::sqrt(1.0f / (float)d);

    const auto kernelFn = [&](int tx, int ty) {
        std::vector<float> row;
        row.reserve(N);

        for (int iterV = 0; iterV < N; iterV++) {
            float qkAcc = 0;
            for (int iterQK = 0; iterQK < d; iterQK++) {
                qkAcc += srcMatQ[ty * d + iterQK] * srcMatK[iterV * d + iterQK];
            }
            row.push_back(qkAcc * invSqrtD);
        }

        const float maxQK = *rgs::max_element(row);
        float divisorAcc = 0;
        for (float qk : row) {
            divisorAcc += std::exp(qk - maxQK);
        }
        for (float& qk : row) {
            qk = std::exp(qk - maxQK) / divisorAcc;
        }

        float o = 0;
        for (int iterV = 0; iterV < N; iterV++) {
            o += row[iterV] * srcMatV[iterV * d + tx];
        }
        dstMatO[ty * d + tx] = o;
    };

    for (int dstX = 0; dstX < d; dstX++) {
        for (int dstY = 0; dstY < N; dstY++) {
            kernelFn(dstX, dstY);
        }
    }
}

TEST_CASE("CPU-FlashAttention-2", "") {
    constexpr vkc::Extent extent{2, 3, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentDst{extent.width(), extent.height(), vk::Format::eR32Sfloat};

    // Src data
    vkc::StbImageBox srcMatQKV = vkc::StbImageBox::createWithExtent(extent) | unwrap;
    std::span<float> srcSpanQKV = std::span{(float*)srcMatQKV.getPData(), extent.elemCount()};
    // srcMatQ = [[0.1,0.2],[0.3,0.4],[0.5,0.6]]
    for (int i = 0; i < srcSpanQKV.size(); i++) {
        srcSpanQKV[i] = (float)(i + 1) / 10.f;
    }

    // CPU Reference
    vkc::StbImageBox dstMatCpuRef = vkc::StbImageBox::createWithExtent(srcMatQKV.getExtent()) | unwrap;
    std::span<float> dstSpan = std::span{(float*)dstMatCpuRef.getPData(), extentDst.elemCount()};

    fa2RefImpl(srcSpanQKV, srcSpanQKV, srcSpanQKV, dstSpan, extent);
    REQUIRE(dstSpan[0] == Catch::Approx(0.305655122f));
    REQUIRE(dstSpan[1] == Catch::Approx(0.405655146f));
}

TEST_CASE("GLSL-FlashAttention-2", "") {
    vkc::initVulkan() | unwrap;

    constexpr float maxValidDiff = 0.0001f;
    constexpr float maxValidAvgDiff = 0.000001f;

    constexpr int N = 128;
    constexpr int d = 32;
    constexpr vkc::Extent extent{d, N, vk::Format::eR32Sfloat};

    // Src data
    vkc::StbImageBox srcMatQ = vkc::StbImageBox::createWithExtent(extent) | unwrap;
    std::span<float> srcSpanQ = std::span{(float*)srcMatQ.getPData(), extent.elemCount()};
    vkc::StbImageBox srcMatK = vkc::StbImageBox::createWithExtent(extent) | unwrap;
    std::span<float> srcSpanK = std::span{(float*)srcMatK.getPData(), extent.elemCount()};
    vkc::StbImageBox srcMatV = vkc::StbImageBox::createWithExtent(extent) | unwrap;
    std::span<float> srcSpanV = std::span{(float*)srcMatV.getPData(), extent.elemCount()};
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (auto& val : srcSpanQ) {
        val = dist(rdEngine);
    }
    for (auto& val : srcSpanK) {
        val = dist(rdEngine);
    }
    for (auto& val : srcSpanV) {
        val = dist(rdEngine);
    }

    // CPU Reference
    vkc::StbImageBox dstMatCpuRef = vkc::StbImageBox::createWithExtent(extent) | unwrap;
    std::span<float> dstMatCpuRefSpan = std::span{(float*)dstMatCpuRef.getPData(), extent.elemCount()};

    fa2RefImpl(srcSpanQ, srcSpanK, srcSpanV, dstMatCpuRefSpan, extent);

    vkc::StbImageBox dstMatVk = vkc::StbImageBox::createWithExtent(extent) | unwrap;

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
    vkc::StorageImageBox srcMatQBox =
        vkc::StorageImageBox::create(pDeviceBox, srcMatQ.getExtent(), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StagingBufferBox srcMatQStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, srcMatQ.getExtent().size(), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StorageImageBox srcMatKBox =
        vkc::StorageImageBox::create(pDeviceBox, srcMatK.getExtent(), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StagingBufferBox srcMatKStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, srcMatK.getExtent().size(), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StorageImageBox srcMatVBox =
        vkc::StorageImageBox::create(pDeviceBox, srcMatV.getExtent(), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StagingBufferBox srcMatVStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, srcMatV.getExtent().size(), vkc::StorageType::ReadOnly) | unwrap;
    const std::array srcMatBoxRefs{std::ref(srcMatQBox), std::ref(srcMatKBox), std::ref(srcMatVBox)};
    vkc::StorageImageBox dstMatBox = vkc::StorageImageBox::create(pDeviceBox, dstMatVk.getExtent()) | unwrap;
    vkc::StagingBufferBox dstMatStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, dstMatVk.getExtent().size(), vkc::StorageType::ReadWrite) | unwrap;
    const std::array dstMatBoxRefs{std::ref(dstMatBox)};
    const std::array dstMatStagingBufferBoxRefs{std::ref(dstMatStagingBufferBox)};
    srcMatQStagingBufferBox.upload(srcMatQ.getPData()) | unwrap;
    srcMatKStagingBufferBox.upload(srcMatK.getPData()) | unwrap;
    srcMatVStagingBufferBox.upload(srcMatV.getPData()) | unwrap;

    const std::vector descPoolSizes = genPoolSizes(srcMatQBox, srcMatKBox, srcMatVBox, dstMatBox);
    vkc::DescPoolBox descPoolBox = vkc::DescPoolBox::create(pDeviceBox, descPoolSizes) | unwrap;

    const std::array fa2DLayoutBindings = genDescSetLayoutBindings(srcMatQBox, srcMatKBox, srcMatVBox, dstMatBox);
    vkc::DescSetLayoutBox fa2DLayoutBox = vkc::DescSetLayoutBox::create(pDeviceBox, fa2DLayoutBindings) | unwrap;
    const std::array fa2DLayoutBoxCRefs{std::cref(fa2DLayoutBox)};
    vkc::PipelineLayoutBox fa2PLayoutBox = vkc::PipelineLayoutBox::create(pDeviceBox, fa2DLayoutBoxCRefs) | unwrap;
    vkc::DescSetsBox fa2DescSetsBox = vkc::DescSetsBox::create(pDeviceBox, descPoolBox, fa2DLayoutBoxCRefs) | unwrap;
    const std::array fa2WriteDescSets = genWriteDescSets(srcMatQBox, srcMatKBox, srcMatVBox, dstMatBox);
    const std::array fa2WriteDescSetss{std::span{fa2WriteDescSets.begin(), fa2WriteDescSets.end()}};
    fa2DescSetsBox.updateDescSets(fa2WriteDescSetss);

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox fa2CmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;

    SECTION("v0") {
        constexpr int groupSizeX = 16;
        constexpr int groupSizeY = 16;
        constexpr int groupNumX = vkc::ceilDiv(extent.width(), groupSizeX);
        constexpr int groupNumY = vkc::ceilDiv(extent.height(), groupSizeY);
        vkc::ShaderBox fa2ShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::flash_attention2::v0::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, groupSizeY};
        vkc::PipelineBox fa2PipelineBox =
            vkc::PipelineBox::createCompute(pDeviceBox, fa2PLayoutBox, fa2ShaderBox, specConstantBox.getSpecInfo()) |
            unwrap;

        fa2CmdBufBox.begin() | unwrap;
        fa2CmdBufBox.bindPipeline(fa2PipelineBox);
        fa2CmdBufBox.bindDescSets(fa2DescSetsBox, fa2PLayoutBox, vk::PipelineBindPoint::eCompute);
        fa2CmdBufBox.recordPrepareReceive<vkc::StorageImageBox>(srcMatBoxRefs);
        fa2CmdBufBox.recordCopyStagingToImage(srcMatQStagingBufferBox, srcMatQBox);
        fa2CmdBufBox.recordCopyStagingToImage(srcMatKStagingBufferBox, srcMatKBox);
        fa2CmdBufBox.recordCopyStagingToImage(srcMatVStagingBufferBox, srcMatVBox);
        fa2CmdBufBox.recordPrepareShaderRead<vkc::StorageImageBox>(srcMatBoxRefs);
        fa2CmdBufBox.recordPrepareShaderWrite(dstMatBoxRefs);
        fa2CmdBufBox.recordDispatch(groupNumX, groupNumY);
        fa2CmdBufBox.recordPrepareSend(dstMatBoxRefs);
        fa2CmdBufBox.recordCopyImageToStaging(dstMatBox, dstMatStagingBufferBox);
        fa2CmdBufBox.recordWaitDownloadComplete(dstMatStagingBufferBoxRefs);
        fa2CmdBufBox.end() | unwrap;

        queueBox.submit(fa2CmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatStagingBufferBox.download(dstMatVk.getPData()) | unwrap;

        float diffAcc = 0;
        std::span<float> dstMatVkSpan = std::span{(float*)dstMatVk.getPData(), extent.elemCount()};
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
        constexpr int groupSizeX = 32;
        constexpr int BrForQ = 16;
        constexpr int BcForKV = 16;
        constexpr int groupNumX = vkc::ceilDiv(extent.height(), BrForQ);
        vkc::ShaderBox fa2ShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::flash_attention2::v1::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, d, BrForQ, BcForKV};
        vkc::PipelineBox fa2PipelineBox =
            vkc::PipelineBox::createCompute(pDeviceBox, fa2PLayoutBox, fa2ShaderBox, specConstantBox.getSpecInfo()) |
            unwrap;

        fa2CmdBufBox.begin() | unwrap;
        fa2CmdBufBox.bindPipeline(fa2PipelineBox);
        fa2CmdBufBox.bindDescSets(fa2DescSetsBox, fa2PLayoutBox, vk::PipelineBindPoint::eCompute);
        fa2CmdBufBox.recordPrepareReceive<vkc::StorageImageBox>(srcMatBoxRefs);
        fa2CmdBufBox.recordCopyStagingToImage(srcMatQStagingBufferBox, srcMatQBox);
        fa2CmdBufBox.recordCopyStagingToImage(srcMatKStagingBufferBox, srcMatKBox);
        fa2CmdBufBox.recordCopyStagingToImage(srcMatVStagingBufferBox, srcMatVBox);
        fa2CmdBufBox.recordPrepareShaderRead<vkc::StorageImageBox>(srcMatBoxRefs);
        fa2CmdBufBox.recordPrepareShaderWrite(dstMatBoxRefs);
        fa2CmdBufBox.recordDispatch(groupNumX, 1);
        fa2CmdBufBox.recordPrepareSend(dstMatBoxRefs);
        fa2CmdBufBox.recordCopyImageToStaging(dstMatBox, dstMatStagingBufferBox);
        fa2CmdBufBox.recordWaitDownloadComplete(dstMatStagingBufferBoxRefs);
        fa2CmdBufBox.end() | unwrap;

        queueBox.submit(fa2CmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatStagingBufferBox.download(dstMatVk.getPData()) | unwrap;

        float diffAcc = 0;
        std::span<float> dstMatVkSpan = std::span{(float*)dstMatVk.getPData(), extent.elemCount()};
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRefSpan, dstMatVkSpan)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstMatVkSpan.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v1 - average diff = {}", avgDiff);
    }
}
