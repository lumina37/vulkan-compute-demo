#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <vector>

#include "spirv/gaussFilter.hpp"
#include "vkc.hpp"
#include "vkc_bin_helper.hpp"

namespace fs = std::filesystem;

int main() {
    vkc::StbImageManager srcImage = vkc::StbImageManager::createFromPath("in.png") | unwrap;
    vkc::StbImageManager dstImage = vkc::StbImageManager::createWithExtent(srcImage.getExtent()) | unwrap;

    // Device
    vkc::DefaultInstanceProps instProps = vkc::DefaultInstanceProps::create() | unwrap;
    if (!instProps.layers.has("VK_LAYER_KHRONOS_validation")) {
        std::println(std::cerr, "VK_LAYER_KHRONOS_validation not supported");
        return -1;
    }
    vkc::InstanceManager instMgr = vkc::InstanceManager::create() | unwrap;
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(instMgr) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.pickDefault() | unwrap).get();
    vkc::PhyDeviceManager& phyDeviceMgr = phyDeviceWithProps.getPhyDeviceMgr();
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceMgr) | unwrap;
    auto pDeviceMgr = std::make_shared<vkc::DeviceManager>(
        vkc::DeviceManager::create(phyDeviceMgr, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}) | unwrap);
    vkc::QueueManager queueMgr = vkc::QueueManager::create(*pDeviceMgr, vk::QueueFlagBits::eCompute) | unwrap;

    // Descriptor & Layouts
    vkc::SamplerManager samplerMgr = vkc::SamplerManager::create(pDeviceMgr) | unwrap;

    constexpr int kernelSize = 23;
    constexpr float sigma = 10.0f;
    vkc::PushConstantManager kernelSizePcMgr{std::pair{kernelSize, sigma * sigma * 2.0f}};

    vkc::ImageManager srcImageMgr =
        vkc::ImageManager::create(phyDeviceMgr, pDeviceMgr, srcImage.getExtent(), vkc::ImageType::Read) | unwrap;
    const std::array srcImageMgrCRefs{std::cref(srcImageMgr)};
    vkc::ImageManager dstImageMgr =
        vkc::ImageManager::create(phyDeviceMgr, pDeviceMgr, srcImage.getExtent(), vkc::ImageType::Write) | unwrap;
    const std::array dstImageMgrCRefs{std::cref(dstImageMgr)};
    srcImageMgr.uploadFrom(srcImage.getImageSpan()) | unwrap;

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
    vkc::TimestampQueryPoolManager queryPoolMgr =
        vkc::TimestampQueryPoolManager::create(pDeviceMgr, 6, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
        unwrap;

    // Pipeline
    constexpr vkc::BlockSize blockSize{16, 16, 1};
    vkc::ShaderManager gaussShaderMgr = vkc::ShaderManager::create(pDeviceMgr, shader::gaussFilterV0SpirvCode) | unwrap;
    vkc::SpecConstantManager specConstantMgr{blockSize.x, blockSize.y};
    vkc::PipelineManager gaussPipelineMgr =
        vkc::PipelineManager::create(pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr, specConstantMgr.getSpecInfo()) |
        unwrap;

    // Gaussian Blur
    for (int i = 0; i < 15; i++) {
        gaussCmdBufMgr.begin() | unwrap;
        gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
        gaussCmdBufMgr.bindDescSets(gaussDescSetsMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.recordResetQueryPool(queryPoolMgr);
        gaussCmdBufMgr.recordSrcPrepareTranfer(srcImageMgrCRefs);
        gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufMgr.recordCopyStagingToSrc(srcImageMgrCRefs);
        gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufMgr.recordSrcPrepareShaderRead(srcImageMgrCRefs);
        gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrCRefs);
        gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        gaussCmdBufMgr.recordDispatch(srcImage.getExtent(), blockSize);
        gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgrCRefs);
        gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufMgr.recordCopyDstToStaging(dstImageMgrCRefs);
        gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgrCRefs);
        gaussCmdBufMgr.end() | unwrap;

        gaussCmdBufMgr.submitTo(queueMgr, fenceMgr) | unwrap;
        fenceMgr.wait() | unwrap;
        fenceMgr.reset() | unwrap;

        auto elapsedTime = queryPoolMgr.getElaspedTimes() | unwrap;
        std::println("============================");
        std::println("Staging to src timecost: {} ms", elapsedTime[0]);
        std::println("Dispatch timecost: {} ms", elapsedTime[1]);
        std::println("Staging to dst timecost: {} ms", elapsedTime[2]);
    }

    dstImageMgr.downloadTo(dstImage.getImageSpan()) | unwrap;
    dstImage.saveTo("out.png") | unwrap;
}
