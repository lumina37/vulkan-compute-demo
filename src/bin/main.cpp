#include <array>
#include <print>
#include <vector>

#include <stb_image.h>
#include <stb_image_write.h>

#include "vkc.hpp"

int main(int argc, char** argv) {
    int width, height, channels;
    constexpr int comps = 1;
    unsigned char* srcImage = stbi_load("in.png", &width, &height, &channels, comps);

    vk::Extent2D extent{(uint32_t)width, (uint32_t)height};

    vkc::InstanceManager instMgr;
    vkc::PhyDeviceManager phyDeviceMgr{instMgr};
    vkc::QueueFamilyManager queueFamilyMgr{instMgr, phyDeviceMgr};
    vkc::DeviceManager deviceMgr{phyDeviceMgr, queueFamilyMgr};
    vkc::QueueManager queueMgr{deviceMgr, queueFamilyMgr};
    vkc::ShaderManager computeShaderMgr{deviceMgr, "../shader/addone.comp.spv"};

    vkc::SamplerManager samplerMgr{deviceMgr};
    vkc::BufferManager bufferMgr{phyDeviceMgr, deviceMgr, extent};
    std::array descSetLayoutBindings =
        genDescSetLayoutBindings(samplerMgr, bufferMgr.srcImageMgr_, bufferMgr.dstImageMgr_);
    vkc::DescPoolManager descPoolMgr{deviceMgr};
    vkc::DescSetLayoutManager descSetLayoutMgr{deviceMgr, descSetLayoutBindings};
    vkc::DescSetManager descSetMgr{deviceMgr, descSetLayoutMgr, descPoolMgr};
    vkc::PipelineLayoutManager pipelineLayoutMgr{deviceMgr, descSetLayoutMgr};

    vkc::PipelineManager pipelineMgr{deviceMgr, pipelineLayoutMgr, extent, computeShaderMgr};
    vkc::CommandPoolManager commandPoolMgr{deviceMgr, queueFamilyMgr};

    vkc::Context context{deviceMgr, commandPoolMgr, pipelineMgr, pipelineLayoutMgr, descSetMgr, queueMgr, extent};

    std::span src{srcImage, (size_t)width * height};
    std::vector<uint8_t> dst(width * height);

    descSetMgr.updateDescSets(samplerMgr, bufferMgr.srcImageMgr_, bufferMgr.dstImageMgr_);

    context.execute(src, dst, bufferMgr);

    stbi_image_free(srcImage);
    stbi_write_png("out.png", width, height, comps, dst.data(), 0);
}
