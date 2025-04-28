#include <array>
#include <cmath>
#include <expected>
#include <filesystem>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <vector>

#include "spirv/gaussFilter.hpp"
#include "vkc.hpp"

namespace fs = std::filesystem;

class Unwrap {
public:
    template <typename T>
    static friend auto operator|(std::expected<T, vkc::Error>&& src, [[maybe_unused]] const Unwrap& _) {
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

int main() {
    vkc::StbImageManager srcImage = vkc::StbImageManager::createFromPath("in.png") | unwrap;
    vkc::StbImageManager dstImage = vkc::StbImageManager::createWithExtent(srcImage.getExtent()) | unwrap;

    // Device
    vkc::InstanceManager instMgr = vkc::InstanceManager::create() | unwrap;
    vkc::PhysicalDeviceManager phyDeviceMgr = vkc::PhysicalDeviceManager::create(instMgr) | unwrap;
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceMgr) | unwrap;
    auto pDeviceMgr =
        std::make_shared<vkc::DeviceManager>(vkc::DeviceManager::create(phyDeviceMgr, computeQFamilyIdx) | unwrap);
    vkc::QueueManager queueMgr = vkc::QueueManager::create(*pDeviceMgr, computeQFamilyIdx) | unwrap;

    // Descriptor & Layouts
    vkc::SamplerManager samplerMgr = vkc::SamplerManager::create(pDeviceMgr) | unwrap;

    constexpr int kernelSize = 23;
    constexpr float sigma = 10.0f;
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
    vkc::TimestampQueryPoolManager queryPoolMgr{pDeviceMgr, 2, phyDeviceMgr.getTimestampPeriod()};

    // Pipeline
    constexpr vkc::BlockSize blockSize{256, 1, 1};
    vkc::ShaderManager gaussShaderMgr{pDeviceMgr, shader::gaussFilterV1SpirvCode};
    constexpr int maxHalfKSize = 128;
    vkc::SpecConstantManager specConstantMgr{blockSize.x, maxHalfKSize};
    vkc::PipelineManager gaussPipelineMgr{pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr, specConstantMgr.getSpecInfo()};

    // Gaussian Blur
    for (int i = 0; i < 15; i++) {
        gaussCmdBufMgr.begin();
        gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
        gaussCmdBufMgr.bindDescSets(gaussDescSetsMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.recordResetQueryPool(queryPoolMgr);
        gaussCmdBufMgr.recordSrcPrepareTranfer(srcImageMgrCRefs);
        gaussCmdBufMgr.recordUploadToSrc(srcImageMgrCRefs);
        gaussCmdBufMgr.recordSrcPrepareShaderRead(srcImageMgrCRefs);
        gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrCRefs);
        gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader);
        gaussCmdBufMgr.recordDispatch(srcImage.getExtent(), blockSize);
        gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader);
        gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgrCRefs);
        gaussCmdBufMgr.recordDownloadToDst(dstImageMgrCRefs);
        gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgrCRefs);
        gaussCmdBufMgr.end();

        gaussCmdBufMgr.submitTo(queueMgr);
        gaussCmdBufMgr.waitFence();

        std::println("Gaussian blur timecost: {} ms", queryPoolMgr.getElaspedTimes()[0]);
    }

    dstImageMgr.downloadTo(dstImage.getImageSpan());
    dstImage.saveTo("out.png") | unwrap;
}
