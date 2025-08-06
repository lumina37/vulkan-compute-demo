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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "spirv/sgemm.hpp"
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
            std::println(std::cerr, "{}:{} cate={} code={} msg={}", fileName, err.source.line(),
                         vkc::errCateToStr(err.cate), err.code, err.msg);
            std::exit(err.code);
        }
        if constexpr (!std::is_void_v<T>) {
            return std::forward_like<T>(src.value());
        }
    }
};

constexpr auto unwrap = Unwrap();

void sgemmRefImpl(const std::span<const float> srcMatA, const std::span<const float> srcMatB,
                  const std::span<float> dstMat, const vkc::Extent extentA, const vkc::Extent extentB) {
    const int AM = extentA.height();
    const int K = extentA.width();
    const int BN = extentB.width();
    for (int dstX = 0; dstX < BN; dstX++) {
        for (int dstY = 0; dstY < AM; dstY++) {
            float acc = 0;
            for (int k = 0; k < K; k++) {
                acc += srcMatA[dstY * K + k] * srcMatB[k * BN + dstX];
            }
            dstMat[dstY * BN + dstX] = acc;
        }
    }
}

TEST_CASE("CPU-SGEMM", "") {
    constexpr vkc::Extent extentA{3, 2, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentB{1, 3, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentDst{extentB.width(), extentA.height(), vk::Format::eR32Sfloat};

    // Src data
    vkc::StbImageBox srcMatA = vkc::StbImageBox::createWithExtent(extentA) | unwrap;
    std::span<float> srcSpanA = std::span{(float*)srcMatA.getPData(), extentA.size() / extentA.bpp()};
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
    constexpr float maxValidDiff = 0.01f;
    constexpr float maxValidAvgDiff = 0.001f;

    constexpr vkc::Extent extentA{32, 16, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentB{64, 32, vk::Format::eR32Sfloat};
    constexpr vkc::Extent extentDst{extentB.width(), extentA.height(), vk::Format::eR32Sfloat};

    // Src data
    vkc::StbImageBox srcMatA = vkc::StbImageBox::createWithExtent(extentA) | unwrap;
    std::span<float> srcSpanA = std::span{(float*)srcMatA.getPData(), extentA.elemCount()};
    vkc::StbImageBox srcMatB = vkc::StbImageBox::createWithExtent(extentB) | unwrap;
    std::span<float> srcSpanB = std::span{(float*)srcMatB.getPData(), extentB.elemCount()};
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    for (auto& val : srcSpanA) {
        val = (float)rdEngine() / std::numeric_limits<uint32_t>::max();
    }
    for (auto& val : srcSpanB) {
        val = (float)rdEngine() / std::numeric_limits<uint32_t>::max();
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

    const std::array gaussDLayoutBindings = genDescSetLayoutBindings(srcMatABox, srcMatBBox, dstMatBox);
    vkc::DescSetLayoutBox gaussDLayoutBox = vkc::DescSetLayoutBox::create(pDeviceBox, gaussDLayoutBindings) | unwrap;
    const std::array gaussDLayoutBoxCRefs{std::cref(gaussDLayoutBox)};
    vkc::PipelineLayoutBox gaussPLayoutBox = vkc::PipelineLayoutBox::create(pDeviceBox, gaussDLayoutBoxCRefs) | unwrap;
    vkc::DescSetsBox gaussDescSetsBox =
        vkc::DescSetsBox::create(pDeviceBox, descPoolBox, gaussDLayoutBoxCRefs) | unwrap;
    const std::array gaussWriteDescSets = genWriteDescSets(srcMatABox, srcMatBBox, dstMatBox);
    const std::array gaussWriteDescSetss{std::span{gaussWriteDescSets.begin(), gaussWriteDescSets.end()}};
    gaussDescSetsBox.updateDescSets(gaussWriteDescSetss);

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox gaussCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;

    SECTION("v0") {
        constexpr vkc::BlockSize blockSize{16, 16, 1};
        vkc::ShaderBox gaussShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::sgemm::v0::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{blockSize.x, blockSize.y};
        vkc::PipelineBox gaussPipelineBox = vkc::PipelineBox::createCompute(pDeviceBox, gaussPLayoutBox, gaussShaderBox,
                                                                            specConstantBox.getSpecInfo()) |
                                            unwrap;

        gaussCmdBufBox.begin() | unwrap;
        gaussCmdBufBox.bindPipeline(gaussPipelineBox);
        gaussCmdBufBox.bindDescSets(gaussDescSetsBox, gaussPLayoutBox, vk::PipelineBindPoint::eCompute);
        gaussCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::StorageImageBox>(srcMatBoxRefs);
        gaussCmdBufBox.recordCopyStagingToSrc(srcMatABox);
        gaussCmdBufBox.recordCopyStagingToSrc(srcMatBBox);
        gaussCmdBufBox.recordSrcPrepareShaderRead<vkc::StorageImageBox>(srcMatBoxRefs);
        gaussCmdBufBox.recordDstPrepareShaderWrite(dstMatBoxRefs);
        gaussCmdBufBox.recordDispatch(extentDst.extent(), blockSize);
        gaussCmdBufBox.recordPrepareSendAfterDispatch(dstMatBoxRefs);
        gaussCmdBufBox.recordCopyDstToStaging(dstMatBox);
        gaussCmdBufBox.recordWaitDownloadComplete(dstMatBoxRefs);
        gaussCmdBufBox.end() | unwrap;

        queueBox.submit(gaussCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstMatBox.download(dstMatVk.getPData()) | unwrap;

        int diffAcc = 0;
        std::span<float> dstMatVkSpan = std::span{(float*)dstMatVk.getPData(), extentDst.elemCount()};
        for (const auto [lhs, rhs] : rgs::views::zip(dstMatCpuRefSpan, dstMatVkSpan)) {
            int diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = (float)diffAcc / (float)dstMatVkSpan.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v0 - average diff = {}", avgDiff);
    }
}
