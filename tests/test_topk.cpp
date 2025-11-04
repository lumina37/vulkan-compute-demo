#include <algorithm>
#include <filesystem>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "spirv/topk.hpp"
#include "vkc.hpp"
#include "vkc_helper.hpp"

namespace fs = std::filesystem;
namespace rgs = std::ranges;

void prefixSumRefImpl(const std::span<const float> srcArr, const std::span<float> dstArr) {
    const int N = (int)srcArr.size();

    std::vector temp(srcArr.begin(), srcArr.end());
    rgs::sort(temp, std::greater{});
    rgs::copy(temp.begin(), temp.begin() + dstArr.size(), dstArr.begin());
}

TEST_CASE("CPU-TopK", "") {
    // Src data
    constexpr int N = 6;
    std::vector<float> srcArr(N);
    // srcArr = [1,2,3,4,5,6]
    for (int i = 0; i < srcArr.size(); i++) {
        srcArr[i] = (float)i + 1;
    }
    // CPU Reference
    std::vector<float> dstArr(2);

    prefixSumRefImpl(srcArr, dstArr);
    REQUIRE(dstArr[0] == Catch::Approx(6.f));
    REQUIRE(dstArr[1] == Catch::Approx(5.f));
}

TEST_CASE("GLSL-TopK", "") {
    vkc::initVulkan() | unwrap;

    constexpr float maxValidDiff = 0.0001f;
    constexpr float maxValidAvgDiff = 0.000001f;

    constexpr int N = 1024;
    constexpr int topk = 4;

    // Src data
    std::vector<float> srcArr(N);
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (auto& val : srcArr) {
        val = dist(rdEngine);
    }

    // CPU Reference
    std::vector<float> dstArrCpuRef(topk);
    prefixSumRefImpl(srcArr, dstArrCpuRef);

    std::vector<float> dstArrVk(topk);

    // Device
    vkc::InstanceBox instBox = vkc::InstanceBox::create() | unwrap;
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(instBox) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    vkc::DefaultPhyDeviceFeatures phyDeviceFeatures = vkc::DefaultPhyDeviceFeatures::create(phyDeviceBox) | unwrap;
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::createWithExts(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}, {},
                                       phyDeviceFeatures.getPFeature()) |
        unwrap);
    vkc::QueueBox queueBox = vkc::QueueBox::create(*pDeviceBox, vk::QueueFlagBits::eCompute) | unwrap;

    // Descriptor & Layouts
    vkc::StorageBufferBox srcArrBox =
        vkc::StorageBufferBox::create(pDeviceBox, srcArr.size(), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StagingBufferBox srcArrStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, srcArrBox.getSize(), vkc::StorageType::ReadOnly) | unwrap;
    const std::array srcMatBoxRefs{std::ref(srcArrBox)};
    vkc::StorageBufferBox dstArrBox =
        vkc::StorageBufferBox::create(pDeviceBox, dstArrVk.size(), vkc::StorageType::ReadWrite) | unwrap;
    vkc::StagingBufferBox dstArrStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, dstArrVk.size(), vkc::StorageType::ReadWrite) | unwrap;
    const std::array dstArrBoxRefs{std::ref(dstArrBox)};
    const std::array dstStagingBufferRefs{std::ref(dstArrStagingBufferBox)};
    srcArrStagingBufferBox.upload((std::byte*)srcArr.data()) | unwrap;

    const std::vector descPoolSizes = genPoolSizes(srcArrBox, dstArrBox);
    vkc::DescPoolBox descPoolBox = vkc::DescPoolBox::create(pDeviceBox, descPoolSizes) | unwrap;

    const std::array topkDLayoutBindings = genDescSetLayoutBindings(srcArrBox, dstArrBox);
    vkc::DescSetLayoutBox topkDLayoutBox = vkc::DescSetLayoutBox::create(pDeviceBox, topkDLayoutBindings) | unwrap;
    const std::array topkDLayoutBoxCRefs{std::cref(topkDLayoutBox)};
    vkc::PipelineLayoutBox topkPLayoutBox = vkc::PipelineLayoutBox::create(pDeviceBox, topkDLayoutBoxCRefs) | unwrap;
    vkc::DescSetsBox topkDescSetsBox = vkc::DescSetsBox::create(pDeviceBox, descPoolBox, topkDLayoutBoxCRefs) | unwrap;
    const std::array topkWriteDescSets = genWriteDescSets(srcArrBox, dstArrBox);
    const std::array topkWriteDescSetss{std::span{topkWriteDescSets.begin(), topkWriteDescSets.end()}};
    topkDescSetsBox.updateDescSets(topkWriteDescSetss);

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox topkCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;

    SECTION("v0") {
        constexpr int groupSizeX = 128;
        vkc::ShaderBox topkShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::topk::v0::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSizeX, 256, topk};
        vkc::PipelineBox topkPipelineBox =
            vkc::PipelineBox::createCompute(pDeviceBox, topkPLayoutBox, topkShaderBox, specConstantBox.getSpecInfo()) |
            unwrap;

        topkCmdBufBox.begin() | unwrap;
        topkCmdBufBox.bindPipeline(topkPipelineBox);
        topkCmdBufBox.bindDescSets(topkDescSetsBox, topkPLayoutBox, vk::PipelineBindPoint::eCompute);
        topkCmdBufBox.recordPrepareReceive<vkc::StorageBufferBox>(srcMatBoxRefs);
        topkCmdBufBox.recordCopyStagingToBuffer(srcArrStagingBufferBox, srcArrBox);
        topkCmdBufBox.recordPrepareShaderRead<vkc::StorageBufferBox>(srcMatBoxRefs);
        topkCmdBufBox.recordPrepareShaderWrite(dstArrBoxRefs);
        topkCmdBufBox.recordDispatch(4, 1);
        topkCmdBufBox.recordPrepareSend(dstArrBoxRefs);
        topkCmdBufBox.recordCopyBufferToStaging(dstArrBox, dstArrStagingBufferBox);
        topkCmdBufBox.recordWaitDownloadComplete(dstStagingBufferRefs);
        topkCmdBufBox.end() | unwrap;

        queueBox.submit(topkCmdBufBox, fenceBox) | unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        dstArrStagingBufferBox.download((std::byte*)dstArrVk.data()) | unwrap;

        float diffAcc = 0;
        for (const auto [lhs, rhs] : rgs::views::zip(dstArrCpuRef, dstArrVk)) {
            const float diff = std::abs(lhs - rhs);
            REQUIRE(diff <= maxValidDiff);
            diffAcc += diff;
        }
        float avgDiff = diffAcc / (float)dstArrVk.size();

        REQUIRE(avgDiff < maxValidAvgDiff);
        std::println("v0 - average diff = {}", avgDiff);
    }
}
