#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "vkc.hpp"

static inline void genGaussKernel(std::span<float> dst, const int kernelSize, const float sigma) {
    const int halfKSize = kernelSize / 2;
    const float doubleSigma2 = 2 * sigma * sigma;

    dst[0] = 1.;
    float sum = dst[0];
    for (int i = 1; i <= halfKSize; i++) {
        const float elem = std::expf((float)(-i * i) / doubleSigma2);
        dst[i] = elem;
        sum += 2 * elem;  // double for both side
    }
    for (auto& elem : dst) {
        elem /= sum;
    }
}

int main(int argc, char** argv) {
    vkc::StbImageManager srcImage{"in.png"};
    vkc::StbImageManager dstImage{srcImage.getExtent()};

    // Device
    vkc::InstanceManager instMgr;
    vkc::PhyDeviceManager phyDeviceMgr{instMgr};
    vkc::QueueFamilyManager queueFamilyMgr{phyDeviceMgr};
    vkc::DeviceManager deviceMgr{phyDeviceMgr, queueFamilyMgr};

    // Descriptor & Layouts
    vkc::SamplerManager samplerMgr{deviceMgr};

    constexpr int uboLen = 16;
    constexpr int maxKernelSize = uboLen * 2 + 1;
    constexpr int kernelSize = 11;
    static_assert(kernelSize <= maxKernelSize);
    vkc::PushConstantManager pushConstantMgr{kernelSize};

    std::array<float, uboLen> weights;
    genGaussKernel(weights, kernelSize, 1.5);
    vkc::UboManager uboManager{phyDeviceMgr, deviceMgr, sizeof(weights)};

    vkc::ImageManager srcImageMgr{phyDeviceMgr, deviceMgr, srcImage.getExtent(),
                                  vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                                  vk::DescriptorType::eSampledImage};

    vkc::ImageManager dstImageMgr{phyDeviceMgr, deviceMgr, srcImage.getExtent(),
                                  vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                                  vk::DescriptorType::eStorageImage};

    std::vector descPoolSizes = genPoolSizes(samplerMgr, srcImageMgr, dstImageMgr, uboManager);
    vkc::DescPoolManager descPoolMgr{deviceMgr, descPoolSizes};
    std::array descSetLayoutBindings = genDescSetLayoutBindings(samplerMgr, srcImageMgr, dstImageMgr, uboManager);
    vkc::DescSetLayoutManager descSetLayoutMgr{deviceMgr, descSetLayoutBindings};
    vkc::PipelineLayoutManager pipelineLayoutMgr{deviceMgr, descSetLayoutMgr, pushConstantMgr.getPushConstantRange()};
    vkc::DescSetManager descSetMgr{deviceMgr, descSetLayoutMgr, descPoolMgr};
    descSetMgr.updateDescSets(samplerMgr, srcImageMgr, dstImageMgr, uboManager);

    // Pipeline
    vkc::ShaderManager computeShaderMgr{deviceMgr, "../shader/gaussianBlur.comp.spv"};
    vkc::PipelineManager pipelineMgr{deviceMgr, pipelineLayoutMgr, computeShaderMgr, {16, 16, 1}};

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
    commandBufferMgr.recordDispatch(srcImage.getExtent());
    commandBufferMgr.recordDownload(dstImageMgr);
    commandBufferMgr.end();

    // Upload Data
    srcImageMgr.uploadFrom(srcImage.getImageSpan());
    uboManager.uploadFrom({(std::byte*)weights.data(), sizeof(weights)});

    // Actual Execution
    commandBufferMgr.submitTo(queueMgr);
    commandBufferMgr.waitFence();

    // Download Data
    dstImageMgr.downloadTo(dstImage.getImageSpan());
    dstImage.saveTo("out.png");
}
