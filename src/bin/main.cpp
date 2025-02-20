#include <array>
#include <vector>

#include <stb_image.h>
#include <stb_image_write.h>

#include "vkc.hpp"

int main(int argc, char** argv) {
    int width, height, oriComps;
    constexpr int comps = 4;
    unsigned char* srcImage = stbi_load("in.png", &width, &height, &oriComps, comps);

    vkc::ExtentManager extent{width, height, 4};

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

    vkc::PipelineManager pipelineMgr{deviceMgr, pipelineLayoutMgr, computeShaderMgr};
    vkc::CommandPoolManager commandPoolMgr{deviceMgr, queueFamilyMgr};

    vkc::Context context{deviceMgr, commandPoolMgr, pipelineMgr, pipelineLayoutMgr, descSetMgr, queueMgr, extent};

    std::span src{srcImage, extent.size()};
    std::vector<uint8_t> dst(extent.size());

    descSetMgr.updateDescSets(samplerMgr, bufferMgr.srcImageMgr_, bufferMgr.dstImageMgr_);

    context.execute(src, dst, bufferMgr);

    stbi_image_free(srcImage);
    stbi_write_png("out.png", width, height, comps, dst.data(), 0);
}
