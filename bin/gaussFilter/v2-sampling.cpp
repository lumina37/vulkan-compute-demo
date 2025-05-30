#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <span>
#include <vector>

#include "../vkc_bin_helper.hpp"
#include "shader.hpp"
#include "vkc.hpp"

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
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
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

    vkc::SampledImageManager srcImageMgr =
        vkc::SampledImageManager::create(phyDeviceMgr, pDeviceMgr, srcImage.getExtent()) | unwrap;
    const std::array srcImageMgrRefs{std::ref(srcImageMgr)};
    vkc::StorageImageManager dstImageMgr =
        vkc::StorageImageManager::create(phyDeviceMgr, pDeviceMgr, srcImage.getExtent()) | unwrap;
    const std::array dstImageMgrRefs{std::ref(dstImageMgr)};

    Timer uploadTimer;
    uploadTimer.begin();
    srcImageMgr.upload(srcImage.getPData()) | unwrap;
    uploadTimer.end();
    std::println("Upload to staging timecost: {} ms", uploadTimer.durationMs());

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
    vkc::CommandBufferManager gaussCmdBufMgr =
        vkc::CommandBufferManager::create(pDeviceMgr, pCommandPoolMgr) | unwrap;
    vkc::TimestampQueryPoolManager queryPoolMgr =
        vkc::TimestampQueryPoolManager::create(pDeviceMgr, 6, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
        unwrap;

    // Pipeline
    constexpr vkc::BlockSize blockSize{16, 16, 1};
    vkc::ShaderManager gaussShaderMgr = vkc::ShaderManager::create(pDeviceMgr, shader::gaussFilter::v0::code) | unwrap;
    vkc::SpecConstantManager specConstantMgr{blockSize.x, blockSize.y};
    vkc::PipelineManager gaussPipelineMgr =
        vkc::PipelineManager::createCompute(pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr,
                                            specConstantMgr.getSpecInfo()) |
        unwrap;

    // Gaussian Blur
    for (int i = 0; i < 15; i++) {
        gaussCmdBufMgr.begin() | unwrap;
        gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
        gaussCmdBufMgr.bindDescSets(gaussDescSetsMgr, gaussPLayoutMgr, vk::PipelineBindPoint::eCompute);
        gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
        gaussCmdBufMgr.recordResetQueryPool(queryPoolMgr);
        gaussCmdBufMgr.recordSrcPrepareTranfer<vkc::SampledImageManager>(srcImageMgrRefs);
        gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufMgr.recordCopyStagingToSrc(srcImageMgr);
        gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufMgr.recordSrcPrepareShaderRead<vkc::SampledImageManager>(srcImageMgrRefs);
        gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrRefs);
        gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        gaussCmdBufMgr.recordDispatch(srcImage.getExtent().extent(), blockSize);
        gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        gaussCmdBufMgr.recordDstPrepareTransfer(dstImageMgrRefs);
        gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufMgr.recordCopyDstToStaging(dstImageMgr);
        gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        gaussCmdBufMgr.recordWaitDownloadComplete(dstImageMgrRefs);
        gaussCmdBufMgr.end() | unwrap;

        gaussCmdBufMgr.submit(queueMgr, fenceMgr) | unwrap;
        fenceMgr.wait() | unwrap;
        fenceMgr.reset() | unwrap;

        auto elapsedTime = queryPoolMgr.getElaspedTimes() | unwrap;
        std::println("============================");
        std::println("Staging to src timecost: {} ms", elapsedTime[0]);
        std::println("Dispatch timecost: {} ms", elapsedTime[1]);
        std::println("Dst from staging timecost: {} ms", elapsedTime[2]);
    }

    Timer downloadTimer;
    downloadTimer.begin();
    dstImageMgr.download(dstImage.getPData()) | unwrap;
    downloadTimer.end();
    std::println("Download from staging timecost: {} ms", downloadTimer.durationMs());

    dstImage.saveTo("out.png") | unwrap;
}
