#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <span>

#include <catch2/catch_test_macros.hpp>

#include "spirv/gaussFilter.hpp"
#include "vkc.hpp"

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

TEST_CASE("Gaussian Blur", "hlsl::gaussFilterVx") {
    constexpr int maxValidDiff = 1;
    constexpr float maxValidAvgDiff = 0.0001;

    constexpr int kernelSize = 5;
    constexpr float sigma = 10.0f;
    constexpr vkc::Extent extent{256, 256, vk::Format::eR8G8B8A8Unorm};

    // Src data
    vkc::StbImageManager srcImage{extent};
    std::default_random_engine rdEngine;
    rdEngine.seed(37);
    for (auto& val : srcImage.getImageSpan()) {
        val = (std::byte)(rdEngine() % std::numeric_limits<uint8_t>::max());
    }

    // CPU Reference
    vkc::StbImageManager dstImageCpuRef{srcImage.getExtent()};
    gaussianFilterRefImpl(srcImage.getImageSpan(), dstImageCpuRef.getImageSpan(), srcImage.getExtent(), kernelSize,
                          sigma);
    dstImageCpuRef.saveTo("ref.png");

    vkc::StbImageManager dstImageVk{srcImage.getExtent()};

    // Device
    vkc::InstanceManager instMgr;
    vkc::PhysicalDeviceManager phyDeviceMgr{instMgr};
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceMgr);
    auto pDeviceMgr = std::make_shared<vkc::DeviceManager>(phyDeviceMgr, computeQFamilyIdx);
    vkc::QueueManager queueMgr{*pDeviceMgr, computeQFamilyIdx};

    // Descriptor & Layouts
    vkc::SamplerManager samplerMgr{pDeviceMgr};
    vkc::PushConstantManager kernelSizePcMgr{std::pair{kernelSize, sigma * sigma * 2.0f}};
    vkc::ImageManager srcImageMgr{phyDeviceMgr, pDeviceMgr, srcImage.getExtent(), vkc::ImageType::Read};
    const std::array srcImageMgrCRefs{std::cref(srcImageMgr)};
    vkc::ImageManager dstImageMgr{phyDeviceMgr, pDeviceMgr, srcImage.getExtent(), vkc::ImageType::Write};
    const std::array dstImageMgrCRefs{std::cref(dstImageMgr)};
    srcImageMgr.uploadFrom(srcImage.getImageSpan());

    const std::vector descPoolSizes = genPoolSizes(srcImageMgr, samplerMgr, dstImageMgr);
    vkc::DescPoolManager descPoolMgr{pDeviceMgr, descPoolSizes};

    const std::array gaussDLayoutBindings = genDescSetLayoutBindings(srcImageMgr, samplerMgr, dstImageMgr);
    vkc::DescSetLayoutManager gaussDLayoutMgr{pDeviceMgr, gaussDLayoutBindings};
    const std::array gaussDLayoutMgrs{std::cref(gaussDLayoutMgr)};
    vkc::PipelineLayoutManager gaussPLayoutMgr{pDeviceMgr, gaussDLayoutMgrs, kernelSizePcMgr.getPushConstantRange()};
    vkc::DescSetsManager gaussDescSetsMgr{pDeviceMgr, descPoolMgr, gaussDLayoutMgrs};
    const std::array gaussWriteDescSets = genWriteDescSets(srcImageMgr, samplerMgr, dstImageMgr);
    const std::array gaussWriteDescSetss{std::span{gaussWriteDescSets.begin(), gaussWriteDescSets.end()}};
    gaussDescSetsMgr.updateDescSets(gaussWriteDescSetss);

    // Command Buffer
    auto pCommandPoolMgr = std::make_shared<vkc::CommandPoolManager>(pDeviceMgr, computeQFamilyIdx);
    vkc::CommandBufferManager gaussCmdBufMgr{pDeviceMgr, pCommandPoolMgr};

    SECTION("v0") {
        constexpr vkc::BlockSize blockSize{16, 16, 1};
        vkc::ShaderManager gaussShaderMgr{pDeviceMgr, shader::gaussFilterV0SpirvCode};
        vkc::PipelineManager gaussPipelineMgr{pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr};

        gaussCmdBufMgr.begin();
        gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
        gaussCmdBufMgr.bindDescSets(gaussDescSetsMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.recordSrcPrepareTranfer(srcImageMgrCRefs);
        gaussCmdBufMgr.recordUploadToSrc(srcImageMgrCRefs);
        gaussCmdBufMgr.recordSrcPrepareShaderRead(srcImageMgrCRefs);
        gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrCRefs);
        gaussCmdBufMgr.recordDispatch(srcImage.getExtent(), blockSize);
        gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgrCRefs);
        gaussCmdBufMgr.recordDownloadToDst(dstImageMgrCRefs);
        gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgrCRefs);
        gaussCmdBufMgr.end();

        gaussCmdBufMgr.submitTo(queueMgr);
        gaussCmdBufMgr.waitFence();

        dstImageMgr.downloadTo(dstImageVk.getImageSpan());
        dstImageVk.saveTo("v0.png");

        int diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstImageCpuRef.getImageSpan(), dstImageVk.getImageSpan())) {
            int diff = std::abs((int)lhs - (int)rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff =
            (float)diffAcc / (float)dstImageVk.getExtent().size() / (float)std::numeric_limits<uint8_t>::max();

        REQUIRE(avgDiff < maxValidAvgDiff);
    }

    SECTION("v1") {
        constexpr vkc::BlockSize blockSize{256, 1, 1};
        vkc::ShaderManager gaussShaderMgr{pDeviceMgr, shader::gaussFilterV1SpirvCode};
        vkc::PipelineManager gaussPipelineMgr{pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr};

        gaussCmdBufMgr.begin();
        gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
        gaussCmdBufMgr.bindDescSets(gaussDescSetsMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.recordSrcPrepareTranfer(srcImageMgrCRefs);
        gaussCmdBufMgr.recordUploadToSrc(srcImageMgrCRefs);
        gaussCmdBufMgr.recordSrcPrepareShaderRead(srcImageMgrCRefs);
        gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrCRefs);
        gaussCmdBufMgr.recordDispatch(srcImage.getExtent(), blockSize);
        gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgrCRefs);
        gaussCmdBufMgr.recordDownloadToDst(dstImageMgrCRefs);
        gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgrCRefs);
        gaussCmdBufMgr.end();

        gaussCmdBufMgr.submitTo(queueMgr);
        gaussCmdBufMgr.waitFence();

        dstImageMgr.downloadTo(dstImageVk.getImageSpan());
        dstImageVk.saveTo("v1.png");

        int diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstImageCpuRef.getImageSpan(), dstImageVk.getImageSpan())) {
            int diff = std::abs((int)lhs - (int)rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff =
            (float)diffAcc / (float)dstImageVk.getExtent().size() / (float)std::numeric_limits<uint8_t>::max();

        REQUIRE(avgDiff < maxValidAvgDiff);
    }

    SECTION("v2") {
        constexpr vkc::BlockSize blockSize{256, 1, 1};
        vkc::ShaderManager gaussShaderMgr{pDeviceMgr, shader::gaussFilterV2SpirvCode};
        vkc::PipelineManager gaussPipelineMgr{pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr};

        gaussCmdBufMgr.begin();
        gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
        gaussCmdBufMgr.bindDescSets(gaussDescSetsMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.recordSrcPrepareTranfer(srcImageMgrCRefs);
        gaussCmdBufMgr.recordUploadToSrc(srcImageMgrCRefs);
        gaussCmdBufMgr.recordSrcPrepareShaderRead(srcImageMgrCRefs);
        gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrCRefs);
        gaussCmdBufMgr.recordDispatch(srcImage.getExtent(), blockSize);
        gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgrCRefs);
        gaussCmdBufMgr.recordDownloadToDst(dstImageMgrCRefs);
        gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgrCRefs);
        gaussCmdBufMgr.end();

        gaussCmdBufMgr.submitTo(queueMgr);
        gaussCmdBufMgr.waitFence();

        dstImageMgr.downloadTo(dstImageVk.getImageSpan());
        dstImageVk.saveTo("v2.png");

        int diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstImageCpuRef.getImageSpan(), dstImageVk.getImageSpan())) {
            int diff = std::abs((int)lhs - (int)rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff =
            (float)diffAcc / (float)dstImageVk.getExtent().size() / (float)std::numeric_limits<uint8_t>::max();

        REQUIRE(avgDiff < maxValidAvgDiff);
    }
}
