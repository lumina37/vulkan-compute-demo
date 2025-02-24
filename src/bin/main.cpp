#include <array>
#include <cstddef>
#include <vector>

#include <stb_image.h>
#include <stb_image_write.h>

#include "vkc.hpp"

int main(int argc, char** argv) {
    int width, height, oriComps;
    constexpr int comps = 4;
    auto* srcImage = (std::byte*)stbi_load("in.png", &width, &height, &oriComps, comps);

    vkc::ExtentManager extent{width, height, 4};

    // Device
    vkc::InstanceManager instMgr;
    vkc::PhyDeviceManager phyDeviceMgr{instMgr};
    vkc::QueueFamilyManager queueFamilyMgr{phyDeviceMgr};
    vkc::DeviceManager deviceMgr{phyDeviceMgr, queueFamilyMgr};

    // Descriptor & Layouts
    vkc::SamplerManager samplerMgr{deviceMgr};
    vkc::PushConstantManager pushConstantMgr{23};
    vkc::ImageManager srcImageMgr{phyDeviceMgr, deviceMgr, extent,
                                  vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                                  vk::DescriptorType::eSampledImage};
    vkc::ImageManager dstImageMgr{phyDeviceMgr, deviceMgr, extent,
                                  vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                                  vk::DescriptorType::eStorageImage};
    std::array descSetLayoutBindings = genDescSetLayoutBindings(samplerMgr, srcImageMgr, dstImageMgr);
    vkc::DescPoolManager descPoolMgr{deviceMgr};
    vkc::DescSetLayoutManager descSetLayoutMgr{deviceMgr, descSetLayoutBindings};
    vkc::PipelineLayoutManager pipelineLayoutMgr{deviceMgr, descSetLayoutMgr, pushConstantMgr.getPushConstantRange()};
    vkc::DescSetManager descSetMgr{deviceMgr, descSetLayoutMgr, descPoolMgr};
    descSetMgr.updateDescSets(samplerMgr, srcImageMgr, dstImageMgr);

    // Pipeline
    vkc::ShaderManager computeShaderMgr{deviceMgr, "../shader/boxFilter.comp.spv"};
    vkc::PipelineManager pipelineMgr{deviceMgr, pipelineLayoutMgr, computeShaderMgr};

    // Command Buffer
    vkc::QueueManager queueMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandPoolManager commandPoolMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandBufferManager commandBufferMgr{deviceMgr, commandPoolMgr};

    commandBufferMgr.begin();
    commandBufferMgr.bindPipeline(pipelineMgr);
    commandBufferMgr.bindDescSet(descSetMgr, pipelineLayoutMgr);
    commandBufferMgr.pushConstant(pushConstantMgr, pipelineLayoutMgr);
    commandBufferMgr.recordUpload(srcImageMgr);
    commandBufferMgr.recordDstLayoutTrans(dstImageMgr);
    commandBufferMgr.recordDispatch(extent);
    commandBufferMgr.recordDownload(dstImageMgr);
    commandBufferMgr.end();

    // Upload Data
    std::span src{srcImage, extent.size()};
    srcImageMgr.uploadFrom(src);

    // Actual Execution
    commandBufferMgr.submitTo(queueMgr);
    commandBufferMgr.waitFence();

    // Download Data
    std::vector<std::byte> dst(extent.size());
    dstImageMgr.downloadTo(dst);

    stbi_write_png("out.png", width, height, comps, dst.data(), 0);
    stbi_image_free(srcImage);
}
