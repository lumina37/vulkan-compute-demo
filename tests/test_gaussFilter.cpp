#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <iostream>
#include <limits>
#include <print>
#include <random>
#include <ranges>
#include <span>

#include <catch2/catch_test_macros.hpp>

#include "spirv/gaussFilter.hpp"
#include "vkc.hpp"

namespace fs = std::filesystem;
namespace rgs = std::ranges;

class Unwrap {
public:
    template <typename T>
    friend auto operator|(std::expected<T, vkc::Error>&& src, [[maybe_unused]] const Unwrap& _) {
        if (!src.has_value()) {
            const auto& err = src.error();
            const fs::path filePath{err.source.file_name()};
            const std::string fileName = filePath.filename().string();
            std::println(std::cerr, "{}:{} msg={} code={}", fileName, err.source.line(), err.msg, err.code);
            std::exit(err.code);
        }
        if constexpr (!std::is_void_v<T>) {
            return std::forward_like<T>(src.value());
        }
    }
};

constexpr auto unwrap = Unwrap();

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

TEST_CASE("Gaussian Blur", "glsl::gaussFilter") {
    constexpr int maxValidDiff = 1;
    constexpr float maxValidAvgDiff = 0.001f;

    constexpr int kernelSize = 5;
    constexpr float sigma = 10.0f;
    constexpr vkc::Extent extent{256, 256, vk::Format::eR8G8B8A8Unorm};

    // Src data
    vkc::StbImageManager srcImage = vkc::StbImageManager::createWithExtent(extent) | unwrap;
    std::default_random_engine rdEngine;
    rdEngine.seed(37);
    for (auto& val : srcImage.getImageSpan()) {
        val = (std::byte)(rdEngine() % std::numeric_limits<uint8_t>::max());
    }

    // CPU Reference
    vkc::StbImageManager dstImageCpuRef = vkc::StbImageManager::createWithExtent(srcImage.getExtent()) | unwrap;
    gaussianFilterRefImpl(srcImage.getImageSpan(), dstImageCpuRef.getImageSpan(), srcImage.getExtent(), kernelSize,
                          sigma);
    // dstImageCpuRef.saveTo("ref.png") | unwrap;

    vkc::StbImageManager dstImageVk = vkc::StbImageManager::createWithExtent(srcImage.getExtent()) | unwrap;

    // Device
    vkc::InstanceManager instMgr = vkc::InstanceManager::create() | unwrap;
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(instMgr) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
    vkc::PhyDeviceManager& phyDeviceMgr = phyDeviceWithProps.getPhyDeviceMgr();
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceMgr) | unwrap;
    auto pDeviceMgr = std::make_shared<vkc::DeviceManager>(
        vkc::DeviceManager::create(phyDeviceMgr, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}) | unwrap);
    vkc::QueueManager queueMgr = vkc::QueueManager::create(*pDeviceMgr, vk::QueueFlagBits::eCompute) | unwrap;

    // Descriptor & Layouts
    vkc::SamplerManager samplerMgr = vkc::SamplerManager::create(pDeviceMgr) | unwrap;
    vkc::PushConstantManager kernelSizePcMgr{std::pair{kernelSize, sigma * sigma * 2.0f}};
    vkc::SampledImageManager srcImageMgr =
        vkc::SampledImageManager::create(phyDeviceMgr, pDeviceMgr, srcImage.getExtent()) | unwrap;
    const std::array srcImageMgrRefs{std::ref(srcImageMgr)};
    vkc::StorageImageManager dstImageMgr =
        vkc::StorageImageManager::create(phyDeviceMgr, pDeviceMgr, srcImage.getExtent()) | unwrap;
    const std::array dstImageMgrRefs{std::ref(dstImageMgr)};
    srcImageMgr.upload(srcImage.getPData()) | unwrap;

    const std::vector descPoolSizes = genPoolSizes(srcImageMgr, samplerMgr, dstImageMgr);
    vkc::DescPoolManager descPoolMgr = vkc::DescPoolManager::create(pDeviceMgr, descPoolSizes) | unwrap;

    const std::array gaussDLayoutBindings = genDescSetLayoutBindings(srcImageMgr, samplerMgr, dstImageMgr);
    vkc::DescSetLayoutManager gaussDLayoutMgr =
        vkc::DescSetLayoutManager::create(pDeviceMgr, gaussDLayoutBindings) | unwrap;
    const std::array gaussDLayoutMgrCRefs{std::cref(gaussDLayoutMgr)};
    vkc::PipelineLayoutManager gaussPLayoutMgr =
        vkc::PipelineLayoutManager::createWithPushConstant(pDeviceMgr, gaussDLayoutMgrCRefs,
                                                           kernelSizePcMgr.getPushConstantRange()) |
        unwrap;
    vkc::DescSetsManager gaussDescSetsMgr =
        vkc::DescSetsManager::create(pDeviceMgr, descPoolMgr, gaussDLayoutMgrCRefs) | unwrap;
    const std::array gaussWriteDescSets = genWriteDescSets(srcImageMgr, samplerMgr, dstImageMgr);
    const std::array gaussWriteDescSetss{std::span{gaussWriteDescSets.begin(), gaussWriteDescSets.end()}};
    gaussDescSetsMgr.updateDescSets(gaussWriteDescSetss);

    // Command Buffer
    vkc::FenceManager fenceMgr = vkc::FenceManager::create(pDeviceMgr) | unwrap;
    auto pCommandPoolMgr = std::make_shared<vkc::CommandPoolManager>(
        vkc::CommandPoolManager::create(pDeviceMgr, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferManager gaussCmdBufMgr = vkc::CommandBufferManager::create(pDeviceMgr, pCommandPoolMgr) | unwrap;

    SECTION("v0") {
        constexpr vkc::BlockSize blockSize{16, 16, 1};
        vkc::ShaderManager gaussShaderMgr =
            vkc::ShaderManager::create(pDeviceMgr, shader::gaussFilter::v0::code) | unwrap;
        vkc::SpecConstantManager specConstantMgr{blockSize.x, blockSize.y};
        vkc::PipelineManager gaussPipelineMgr =
            vkc::PipelineManager::createCompute(pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr,
                                                specConstantMgr.getSpecInfo()) |
            unwrap;

        gaussCmdBufMgr.begin() | unwrap;
        gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
        gaussCmdBufMgr.bindDescSets(gaussDescSetsMgr, gaussPLayoutMgr, vk::PipelineBindPoint::eCompute);
        gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.recordSrcPrepareTranfer<vkc::SampledImageManager>(srcImageMgrRefs);
        gaussCmdBufMgr.recordCopyStagingToSrc(srcImageMgr);
        gaussCmdBufMgr.recordSrcPrepareShaderRead<vkc::SampledImageManager>(srcImageMgrRefs);
        gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrRefs);
        gaussCmdBufMgr.recordDispatch(srcImage.getExtent().extent(), blockSize);
        gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgrRefs);
        gaussCmdBufMgr.recordCopyDstToStaging(dstImageMgr);
        gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgrRefs);
        gaussCmdBufMgr.end() | unwrap;

        gaussCmdBufMgr.submit(queueMgr, fenceMgr) | unwrap;
        fenceMgr.wait() | unwrap;
        fenceMgr.reset() | unwrap;

        dstImageMgr.download(dstImageVk.getPData()) | unwrap;
        // dstImageVk.saveTo("v0.png") | unwrap;

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
        vkc::ShaderManager gaussShaderMgr =
            vkc::ShaderManager::create(pDeviceMgr, shader::gaussFilter::v1::code) | unwrap;
        vkc::SpecConstantManager specConstantMgr{blockSize.x};
        vkc::PipelineManager gaussPipelineMgr =
            vkc::PipelineManager::createCompute(pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr,
                                                specConstantMgr.getSpecInfo()) |
            unwrap;

        gaussCmdBufMgr.begin() | unwrap;
        gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
        gaussCmdBufMgr.bindDescSets(gaussDescSetsMgr, gaussPLayoutMgr, vk::PipelineBindPoint::eCompute);
        gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.recordSrcPrepareTranfer<vkc::SampledImageManager>(srcImageMgrRefs);
        gaussCmdBufMgr.recordCopyStagingToSrc(srcImageMgr);
        gaussCmdBufMgr.recordSrcPrepareShaderRead<vkc::SampledImageManager>(srcImageMgrRefs);
        gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrRefs);
        gaussCmdBufMgr.recordDispatch(srcImage.getExtent().extent(), blockSize);
        gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgrRefs);
        gaussCmdBufMgr.recordCopyDstToStaging(dstImageMgr);
        gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgrRefs);
        gaussCmdBufMgr.end() | unwrap;

        gaussCmdBufMgr.submit(queueMgr, fenceMgr) | unwrap;
        fenceMgr.wait() | unwrap;
        fenceMgr.reset() | unwrap;

        dstImageMgr.download(dstImageVk.getPData()) | unwrap;
        // dstImageVk.saveTo("v2.png") | unwrap;

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
