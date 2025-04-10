#include <catch2/catch_test_macros.hpp>

#include "spirv/gaussFilter.hpp"
#include "vkc.hpp"

TEST_CASE("Gaussian Blur", "hlsl::gaussFilterVx") {
    vkc::StbImageManager srcImage{"in.png"};
    vkc::StbImageManager dstImage{srcImage.getExtent()};

    // Device
    vkc::InstanceManager instMgr;
    vkc::PhysicalDeviceManager phyDeviceMgr{instMgr};
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceMgr);
    auto pDeviceMgr = std::make_shared<vkc::DeviceManager>(phyDeviceMgr, computeQFamilyIdx);
    vkc::QueueManager queueMgr{*pDeviceMgr, computeQFamilyIdx};

    // Descriptor & Layouts
    vkc::SamplerManager samplerMgr{pDeviceMgr};

    constexpr int kernelSize = 49;
    constexpr float sigma = 10.0f;
    vkc::PushConstantManager kernelSizePcMgr{std::pair{kernelSize, sigma * sigma * 2.0f}};

    vkc::ImageManager srcImageMgr{phyDeviceMgr, pDeviceMgr, srcImage.getExtent(), vkc::ImageType::Read};
    const std::array srcImageMgrCRefs{std::cref(srcImageMgr)};
    vkc::ImageManager dstImageMgr{phyDeviceMgr, pDeviceMgr, srcImage.getExtent(), vkc::ImageType::Write};
    const std::array dstImageMgrCRefs{std::cref(dstImageMgr)};

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

    // Pipeline
    constexpr vkc::BlockSize blockSize{16, 16, 1};
    vkc::ShaderManager gaussShaderMgr{pDeviceMgr, shader::gaussFilterV0SpirvCode};
    vkc::PipelineManager gaussPipelineMgr{pDeviceMgr, gaussPLayoutMgr, gaussShaderMgr};

    // Command Buffer
    auto pCommandPoolMgr = std::make_shared<vkc::CommandPoolManager>(pDeviceMgr, computeQFamilyIdx);
    vkc::CommandBufferManager gaussCmdBufMgr{pDeviceMgr, pCommandPoolMgr};

    // Gaussian Blur
    srcImageMgr.uploadFrom(srcImage.getImageSpan());

    for (int i = 0; i < 15; i++) {
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
    }

    dstImageMgr.downloadTo(dstImage.getImageSpan());


}
