#include <print>

#include "vkc.hpp"

int main(int argc, char** argv) {
    vk::Extent2D extent{3, 2};

    vkc::InstanceManager instMgr;
    vkc::PhyDeviceManager phyDeviceMgr{instMgr};
    vkc::QueueFamilyManager queueFamilyMgr{instMgr, phyDeviceMgr};
    vkc::DeviceManager deviceMgr{phyDeviceMgr, queueFamilyMgr};
    vkc::QueueManager queueMgr{deviceMgr, queueFamilyMgr};
    vkc::ShaderManager computeShaderMgr{deviceMgr, "../shader/addone.comp.spv"};
    vkc::DescPoolManager descPoolMgr{deviceMgr};
    vkc::DescSetLayoutManager descSetLayoutMgr{deviceMgr};
    vkc::DescSetManager descSetMgr{deviceMgr, descSetLayoutMgr, descPoolMgr};
    vkc::PipelineLayoutManager pipelineLayoutMgr{deviceMgr, descSetLayoutMgr};
    vkc::PipelineManager pipelineMgr{deviceMgr, pipelineLayoutMgr, extent, computeShaderMgr};
    vkc::CommandPoolManager commandPoolMgr{deviceMgr, queueFamilyMgr};

    vkc::Context context{deviceMgr, commandPoolMgr, pipelineMgr, pipelineLayoutMgr, descSetMgr, queueMgr, extent};

    const std::vector<uint8_t> src{1, 2, 3, 4, 5, 6};
    std::vector<uint8_t> dst(src.size());
    vkc::BufferManager bufferMgr{phyDeviceMgr, deviceMgr, extent};
    descSetMgr.updateDescSets(bufferMgr.srcImageMgr_, bufferMgr.dstImageMgr_);

    context.execute(src, dst, bufferMgr);

    std::println("src: {}", src);
    std::println("dst: {}", dst);
}
