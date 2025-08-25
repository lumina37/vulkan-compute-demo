#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <limits>
#include <print>
#include <random>
#include <ranges>
#include <span>

#include <catch2/catch_test_macros.hpp>

#include "spirv/gaussFilter.hpp"
#include "vkc.hpp"
#include "vkc_helper.hpp"

namespace fs = std::filesystem;
namespace rgs = std::ranges;

void gaussianFilterRefImpl(const std::span<const std::byte> src, const std::span<std::byte> dst,
                           const vkc::Extent extent, const int kernelSize, const float sigma) {
    const auto getPix = [&extent](auto* pBase, int x, int y, int compIdx) {
        if (x < 0) x = std::abs(x) - 1;
        if (y < 0) y = std::abs(y) - 1;
        if (x >= extent.width()) x = extent.width() * 2 - x - 1;
        if (y >= extent.height()) y = extent.height() * 2 - y - 1;

        const size_t offset = y * extent.rowPitch() + x * extent.bpp() + compIdx;
        assert(offset < extent.size());

        return pBase + offset;
    };

    const int halfKSize = kernelSize / 2;
    const float sigma2 = sigma * sigma * 2.0f;

    std::vector<float> colors(extent.bpp());
    // For each pixel
    for (int row = 0; row < extent.height(); row++) {
        for (int col = 0; col < extent.width(); col++) {
            // Init stats
            for (auto& color : colors) color = 0.0f;
            float weightSum = 0.0f;
            // For each pixel in window
            for (int y = -halfKSize; y <= halfKSize; y++) {
                for (int x = -halfKSize; x <= halfKSize; x++) {
                    const float weight = std::exp(-float(x * x + y * y) / sigma2);
                    for (int compIdx = 0; compIdx < extent.bpp(); compIdx++) {
                        const uint8_t* pSrcVal = (uint8_t*)getPix(src.data(), col + x, row + y, compIdx);
                        colors[compIdx] += (float)(*pSrcVal) * weight;
                    }
                    weightSum += weight;
                }
            }
            // Write to dst
            for (const auto [compIdx, color] : rgs::views::enumerate(colors)) {
                uint8_t* pDstVal = (uint8_t*)getPix(dst.data(), col, row, (int)compIdx);
                *pDstVal = (uint8_t)std::round(color / weightSum);
            }
            // Keep alpha
            if (extent.bpp() == 4) {
                uint8_t* pDstVal = (uint8_t*)getPix(dst.data(), col, row, 3);
                *pDstVal = std::numeric_limits<uint8_t>::max();
            }
        }
    }
}

TEST_CASE("GLSL-Gaussian-Blur", "") {
    vkc::initVulkan() | unwrap;

    constexpr int maxValidDiff = 1;
    constexpr float maxValidAvgDiff = 0.001f;

    constexpr int kernelSize = 5;
    constexpr float sigma = 10.0f;
    constexpr vkc::Extent extent{256, 256, vk::Format::eR8G8B8A8Unorm};

    // Src data
    vkc::StbImageBox srcImage = vkc::StbImageBox::createWithExtent(extent) | unwrap;
    std::default_random_engine rdEngine;
    rdEngine.seed(37);
    for (auto& val : srcImage.getImageSpan()) {
        val = (std::byte)(rdEngine() % std::numeric_limits<uint8_t>::max());
    }

    // CPU Reference
    vkc::StbImageBox dstImageCpuRef = vkc::StbImageBox::createWithExtent(srcImage.getExtent()) | unwrap;
    gaussianFilterRefImpl(srcImage.getImageSpan(), dstImageCpuRef.getImageSpan(), srcImage.getExtent(), kernelSize,
                          sigma);
    // dstImageCpuRef.saveTo("ref.png") | unwrap;

    vkc::StbImageBox dstImageVk = vkc::StbImageBox::createWithExtent(srcImage.getExtent()) | unwrap;

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
    vkc::SamplerBox samplerBox = vkc::SamplerBox::create(pDeviceBox) | unwrap;
    vkc::PushConstantBox kernelSizePcBox{std::pair{kernelSize, sigma * sigma * 2.0f}};
    vkc::SampledImageBox srcImageBox =
        vkc::SampledImageBox::create(phyDeviceBox, pDeviceBox, srcImage.getExtent()) | unwrap;
    const std::array srcImageBoxRefs{std::ref(srcImageBox)};
    vkc::StorageImageBox dstImageBox =
        vkc::StorageImageBox::create(phyDeviceBox, pDeviceBox, srcImage.getExtent()) | unwrap;
    const std::array dstImageBoxRefs{std::ref(dstImageBox)};
    srcImageBox.upload(srcImage.getPData()) | unwrap;

    const std::vector descPoolSizes = genPoolSizes(srcImageBox, samplerBox, dstImageBox);
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

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox gaussCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;

    SECTION("v0") {
        constexpr int groupSizeX = 16;
        constexpr int groupSizeY = 16;
        const int groupNumX = vkc::ceilDiv(dstImageVk.getExtent().width(), groupSizeX);
        const int groupNumY = vkc::ceilDiv(dstImageVk.getExtent().height(), groupSizeY);
        vkc::ShaderBox gaussShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::gaussFilter::v0::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, groupSizeY};
        vkc::PipelineBox gaussPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, gaussPLayoutBox, gaussShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        gaussCmdBufBox.begin() | unwrap;
        gaussCmdBufBox.bindPipeline(gaussPipelineBox);
        gaussCmdBufBox.bindDescSets(gaussDescSetsBox, gaussPLayoutBox, vk::PipelineBindPoint::eCompute);
        gaussCmdBufBox.pushConstant(kernelSizePcBox, gaussPLayoutBox);
        gaussCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::SampledImageBox>(srcImageBoxRefs);
        gaussCmdBufBox.recordCopyStagingToSrc(srcImageBox);
        gaussCmdBufBox.recordSrcPrepareShaderRead<vkc::SampledImageBox>(srcImageBoxRefs);
        gaussCmdBufBox.recordDstPrepareShaderWrite(dstImageBoxRefs);
        gaussCmdBufBox.recordDispatch(groupNumX, groupNumY);
        gaussCmdBufBox.recordPrepareSendAfterDispatch(dstImageBoxRefs);
        gaussCmdBufBox.recordCopyDstToStaging(dstImageBox);
        gaussCmdBufBox.recordWaitDownloadComplete(dstImageBoxRefs);
        gaussCmdBufBox.end() | unwrap;

        queueBox.submit(gaussCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstImageBox.download(dstImageVk.getPData()) | unwrap;
        // dstImageVk.saveTo("v0.png") | unwrap;

        int diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstImageCpuRef.getImageSpan(), dstImageVk.getImageSpan())) {
            const int diff = std::abs((int)lhs - (int)rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        const float avgDiff =
            (float)diffAcc / (float)dstImageVk.getExtent().size() / (float)std::numeric_limits<uint8_t>::max();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v0 - average diff = {}", avgDiff);
    }

    SECTION("v1") {
        constexpr int groupSizeX = 256;
        const int groupNumX = vkc::ceilDiv(dstImageVk.getExtent().width(), groupSizeX);
        const int groupNumY = dstImageVk.getExtent().height();
        vkc::ShaderBox gaussShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::gaussFilter::v1::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX};
        vkc::PipelineBox gaussPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, gaussPLayoutBox, gaussShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        gaussCmdBufBox.begin() | unwrap;
        gaussCmdBufBox.bindPipeline(gaussPipelineBox);
        gaussCmdBufBox.bindDescSets(gaussDescSetsBox, gaussPLayoutBox, vk::PipelineBindPoint::eCompute);
        gaussCmdBufBox.pushConstant(kernelSizePcBox, gaussPLayoutBox);
        gaussCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::SampledImageBox>(srcImageBoxRefs);
        gaussCmdBufBox.recordCopyStagingToSrc(srcImageBox);
        gaussCmdBufBox.recordSrcPrepareShaderRead<vkc::SampledImageBox>(srcImageBoxRefs);
        gaussCmdBufBox.recordDstPrepareShaderWrite(dstImageBoxRefs);
        gaussCmdBufBox.recordDispatch(groupNumX, groupNumY);
        gaussCmdBufBox.recordPrepareSendAfterDispatch(dstImageBoxRefs);
        gaussCmdBufBox.recordCopyDstToStaging(dstImageBox);
        gaussCmdBufBox.recordWaitDownloadComplete(dstImageBoxRefs);
        gaussCmdBufBox.end() | unwrap;

        queueBox.submit(gaussCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstImageBox.download(dstImageVk.getPData()) | unwrap;
        // dstImageVk.saveTo("v2.png") | unwrap;

        int diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstImageCpuRef.getImageSpan(), dstImageVk.getImageSpan())) {
            const int diff = std::abs((int)lhs - (int)rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        const float avgDiff =
            (float)diffAcc / (float)dstImageVk.getExtent().size() / (float)std::numeric_limits<uint8_t>::max();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v1 - average diff = {}", avgDiff);
    }
}
