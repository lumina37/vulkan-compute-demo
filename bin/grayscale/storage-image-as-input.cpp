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
    vkc::StorageImageManager srcImageMgr =
        vkc::StorageImageManager::create(phyDeviceMgr, pDeviceMgr, srcImage.getExtent(), vkc::StorageImageType::Read) |
        unwrap;
    const std::array srcImageMgrRefs{std::ref(srcImageMgr)};
    vkc::StorageImageManager dstImageMgr =
        vkc::StorageImageManager::create(phyDeviceMgr, pDeviceMgr, srcImage.getExtent()) | unwrap;
    const std::array dstImageMgrRefs{std::ref(dstImageMgr)};

    Timer uploadTimer;
    uploadTimer.begin();
    srcImageMgr.upload(srcImage.getPData()) | unwrap;
    uploadTimer.end();
    std::println("Upload to staging timecost: {} ms", uploadTimer.durationMs());

    const std::vector descPoolSizes = genPoolSizes(srcImageMgr, dstImageMgr);
    vkc::DescPoolManager descPoolMgr = vkc::DescPoolManager::create(pDeviceMgr, descPoolSizes) | unwrap;

    const std::array grayDLayoutBindings = genDescSetLayoutBindings(srcImageMgr, dstImageMgr);
    vkc::DescSetLayoutManager grayDLayoutMgr =
        vkc::DescSetLayoutManager::create(pDeviceMgr, grayDLayoutBindings) | unwrap;
    const std::array grayDLayoutMgrCRefs{std::cref(grayDLayoutMgr)};
    vkc::PipelineLayoutManager grayPLayoutMgr =
        vkc::PipelineLayoutManager::create(pDeviceMgr, grayDLayoutMgrCRefs) | unwrap;
    vkc::DescSetsManager grayDescSetsMgr =
        vkc::DescSetsManager::create(pDeviceMgr, descPoolMgr, grayDLayoutMgrCRefs) | unwrap;
    const std::array grayWriteDescSets = genWriteDescSets(srcImageMgr, dstImageMgr);
    const std::array grayWriteDescSetss{std::span{grayWriteDescSets.begin(), grayWriteDescSets.end()}};
    grayDescSetsMgr.updateDescSets(grayWriteDescSetss);

    // Command Buffer
    vkc::FenceManager fenceMgr = vkc::FenceManager::create(pDeviceMgr) | unwrap;
    auto pCommandPoolMgr = std::make_shared<vkc::CommandPoolManager>(
        vkc::CommandPoolManager::create(pDeviceMgr, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferManager grayCmdBufMgr = vkc::CommandBufferManager::create(pDeviceMgr, pCommandPoolMgr) | unwrap;
    vkc::TimestampQueryPoolManager queryPoolMgr =
        vkc::TimestampQueryPoolManager::create(pDeviceMgr, 6, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
        unwrap;

    // Pipeline
    constexpr vkc::BlockSize blockSize{16, 16, 1};
    vkc::ShaderManager grayShaderMgr = vkc::ShaderManager::create(pDeviceMgr, shader::grayscale::rw::code) | unwrap;
    vkc::SpecConstantManager specConstantMgr{blockSize.x, blockSize.y};
    vkc::PipelineManager grayPipelineMgr =
        vkc::PipelineManager::createCompute(pDeviceMgr, grayPLayoutMgr, grayShaderMgr, specConstantMgr.getSpecInfo()) | unwrap;

    // Gaussian Blur
    for (int i = 0; i < 15; i++) {
        grayCmdBufMgr.begin() | unwrap;
        grayCmdBufMgr.bindPipeline(grayPipelineMgr);
        grayCmdBufMgr.bindDescSets(grayDescSetsMgr, grayPLayoutMgr, vk::PipelineBindPoint::eCompute);
        grayCmdBufMgr.recordResetQueryPool(queryPoolMgr);
        grayCmdBufMgr.recordSrcPrepareTranfer<vkc::StorageImageManager>(srcImageMgrRefs);
        grayCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        grayCmdBufMgr.recordCopyStagingToSrc(srcImageMgr);
        grayCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        grayCmdBufMgr.recordSrcPrepareShaderRead<vkc::StorageImageManager>(srcImageMgrRefs);
        grayCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrRefs);
        grayCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        grayCmdBufMgr.recordDispatch(srcImage.getExtent().extent(), blockSize);
        grayCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
        grayCmdBufMgr.recordDstPrepareTransfer(dstImageMgrRefs);
        grayCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        grayCmdBufMgr.recordCopyDstToStaging(dstImageMgr);
        grayCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
        grayCmdBufMgr.recordWaitDownloadComplete(dstImageMgrRefs);
        grayCmdBufMgr.end() | unwrap;

        grayCmdBufMgr.submit(queueMgr, fenceMgr) | unwrap;
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
