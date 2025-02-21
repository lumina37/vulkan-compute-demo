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
    vkc::BufferManager bufferMgr{phyDeviceMgr, deviceMgr, extent};
    std::array descSetLayoutBindings =
        genDescSetLayoutBindings(samplerMgr, bufferMgr.getSrcImageMgr(), bufferMgr.getDstImageMgr());
    vkc::DescPoolManager descPoolMgr{deviceMgr};
    vkc::DescSetLayoutManager descSetLayoutMgr{deviceMgr, descSetLayoutBindings};
    vkc::PipelineLayoutManager pipelineLayoutMgr{deviceMgr, descSetLayoutMgr, pushConstantMgr.getPushConstantRange()};
    vkc::DescSetManager descSetMgr{deviceMgr, descSetLayoutMgr, descPoolMgr};
    descSetMgr.updateDescSets(samplerMgr, bufferMgr.getSrcImageMgr(), bufferMgr.getDstImageMgr());

    // Pipeline
    vkc::ShaderManager computeShaderMgr{deviceMgr, "../shader/addone.comp.spv"};
    vkc::PipelineManager pipelineMgr{deviceMgr, pipelineLayoutMgr, computeShaderMgr};

    // Command Buffer
    vkc::QueueManager queueMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandPoolManager commandPoolMgr{deviceMgr, queueFamilyMgr};
    vkc::CommandBufferManager commandBufferMgr{deviceMgr, commandPoolMgr};

    commandBufferMgr.begin();
    commandBufferMgr.bindPipeline(pipelineMgr);
    commandBufferMgr.bindDescSet(descSetMgr, pipelineLayoutMgr);
    commandBufferMgr.pushConstant(pushConstantMgr, pipelineLayoutMgr);
    commandBufferMgr.recordUpload(bufferMgr.getSrcImageMgr());
    commandBufferMgr.recordDstLayoutTrans(bufferMgr.getDstImageMgr());
    commandBufferMgr.recordDispatch(extent);
    commandBufferMgr.recordDownload(bufferMgr.getDstImageMgr());
    commandBufferMgr.end();

    // Upload Data
    std::span src{srcImage, extent.size()};
    bufferMgr.getSrcImageMgr().uploadFrom(src);

    // Actual Execution
    commandBufferMgr.submitTo(queueMgr);
    commandBufferMgr.waitFence();

    // Download Data
    std::vector<std::byte> dst(extent.size());
    bufferMgr.getDstImageMgr().downloadTo(dst);

    stbi_write_png("out.png", width, height, comps, dst.data(), 0);
    stbi_image_free(srcImage);
}
