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
    srcImageMgr.upload(srcImage.getPData()) | unwrap;

    const std::vector descPoolSizes =
        genPoolSizes(srcImageMgr, samplerMgr, dstImageMgr, srcImageMgr, samplerMgr, dstImageMgr);
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

    const std::array grayDLayoutBindings = genDescSetLayoutBindings(srcImageMgr, samplerMgr, dstImageMgr);
    vkc::DescSetLayoutManager grayDLayoutMgr =
        vkc::DescSetLayoutManager::create(pDeviceMgr, grayDLayoutBindings) | unwrap;
    const std::array grayDLayoutMgrCRefs{std::cref(grayDLayoutMgr)};
    vkc::PipelineLayoutManager grayPLayoutMgr =
        vkc::PipelineLayoutManager::create(pDeviceMgr, grayDLayoutMgrCRefs) | unwrap;
    vkc::DescSetsManager grayDescSetsMgr =
        vkc::DescSetsManager::create(pDeviceMgr, descPoolMgr, grayDLayoutMgrCRefs) | unwrap;
    const std::array grayWriteDescSets = genWriteDescSets(srcImageMgr, samplerMgr, dstImageMgr);
    const std::array grayWriteDescSetss{std::span{grayWriteDescSets.begin(), grayWriteDescSets.end()}};
    grayDescSetsMgr.updateDescSets(grayWriteDescSetss);

    // Command Buffer
    vkc::SemaphoreManager semaphoreMgr = vkc::SemaphoreManager::create(pDeviceMgr) | unwrap;
    vkc::FenceManager fenceMgr = vkc::FenceManager::create(pDeviceMgr) | unwrap;
    auto pCommandPoolMgr = std::make_shared<vkc::CommandPoolManager>(
        vkc::CommandPoolManager::create(pDeviceMgr, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferManager gaussCmdBufMgr = vkc::CommandBufferManager::create(pDeviceMgr, pCommandPoolMgr) | unwrap;
    vkc::CommandBufferManager grayCmdBufMgr = vkc::CommandBufferManager::create(pDeviceMgr, pCommandPoolMgr) | unwrap;
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

    vkc::ShaderManager grayShaderMgr = vkc::ShaderManager::create(pDeviceMgr, shader::grayscale::ro::code) | unwrap;
    vkc::PipelineManager grayPipelineMgr =
        vkc::PipelineManager::createCompute(pDeviceMgr, grayPLayoutMgr, grayShaderMgr, specConstantMgr.getSpecInfo()) |
        unwrap;

    // Gaussian Blur
    gaussCmdBufMgr.begin() | unwrap;
    gaussCmdBufMgr.bindPipeline(gaussPipelineMgr);
    gaussCmdBufMgr.bindDescSets(gaussDescSetsMgr, gaussPLayoutMgr, vk::PipelineBindPoint::eCompute);
    gaussCmdBufMgr.pushConstant(kernelSizePcMgr, gaussPLayoutMgr);
    gaussCmdBufMgr.recordResetQueryPool(queryPoolMgr);
    gaussCmdBufMgr.recordPrepareReceiveBeforeDispatch<vkc::SampledImageManager>(srcImageMgrRefs);
    gaussCmdBufMgr.recordCopyStagingToSrc(srcImageMgr);
    gaussCmdBufMgr.recordSrcPrepareShaderRead<vkc::SampledImageManager>(srcImageMgrRefs);
    gaussCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrRefs);
    gaussCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    gaussCmdBufMgr.recordDispatch(srcImage.getExtent().extent(), blockSize);
    gaussCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    gaussCmdBufMgr.end() | unwrap;
    queueMgr.submit(gaussCmdBufMgr, semaphoreMgr) | unwrap;

    // Grayscale
    grayCmdBufMgr.begin() | unwrap;
    grayCmdBufMgr.bindPipeline(grayPipelineMgr);
    grayCmdBufMgr.bindDescSets(grayDescSetsMgr, grayPLayoutMgr, vk::PipelineBindPoint::eCompute);
    grayCmdBufMgr.recordPrepareReceiveBeforeDispatch<vkc::SampledImageManager>(srcImageMgrRefs);
    grayCmdBufMgr.recordPrepareSendBeforeDispatch(dstImageMgrRefs);
    grayCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
    grayCmdBufMgr.recordCopyStorageToAnother(dstImageMgr, srcImageMgr);
    grayCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eTransfer) | unwrap;
    grayCmdBufMgr.recordSrcPrepareShaderRead<vkc::SampledImageManager>(srcImageMgrRefs);
    grayCmdBufMgr.recordDstPrepareShaderWrite(dstImageMgrRefs);
    grayCmdBufMgr.recordTimestampStart(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    grayCmdBufMgr.recordDispatch(srcImage.getExtent().extent(), blockSize);
    grayCmdBufMgr.recordTimestampEnd(queryPoolMgr, vk::PipelineStageFlagBits::eComputeShader) | unwrap;
    grayCmdBufMgr.recordPrepareSendAfterDispatch(dstImageMgrRefs);
    grayCmdBufMgr.recordCopyDstToStaging(dstImageMgr);
    grayCmdBufMgr.recordWaitDownloadComplete(dstImageMgrRefs);
    grayCmdBufMgr.end() | unwrap;

    queueMgr.submitAndWaitSemaphore(grayCmdBufMgr, semaphoreMgr, vk::PipelineStageFlagBits::eTransfer, fenceMgr) |
        unwrap;
    fenceMgr.wait() | unwrap;
    fenceMgr.reset() | unwrap;

    auto elapsedTime = queryPoolMgr.getElaspedTimes() | unwrap;
    std::println("GaussFilter dispatch timecost: {} ms", elapsedTime[0]);
    std::println("Storage to sampled transfer timecost: {} ms", elapsedTime[1]);
    std::println("Grayscale dispatch timecost: {} ms", elapsedTime[2]);

    dstImageMgr.download(dstImage.getPData()) | unwrap;
    dstImage.saveTo("out.png") | unwrap;
}
