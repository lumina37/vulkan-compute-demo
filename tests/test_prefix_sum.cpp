#include <filesystem>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "spirv/prefix_sum.hpp"
#include "vkc.hpp"
#include "vkc_helper.hpp"

namespace fs = std::filesystem;
namespace rgs = std::ranges;

void prefixSumRefImpl(const std::span<const float> srcArr, const std::span<float> dstArr) {
    const int N = (int)srcArr.size();

    float prefixAcc = 0.f;
    for (int i = 0; i < N; i++) {
        prefixAcc += srcArr[i];
        dstArr[i] = prefixAcc;
    }
}

TEST_CASE("CPU-PrefixSum", "") {
    // Src data
    constexpr int N = 6;
    std::vector<float> srcArr(N);
    // srcArr = [1,2,3,4,5,6]
    for (int i = 0; i < srcArr.size(); i++) {
        srcArr[i] = (float)i + 1;
    }
    // CPU Reference
    std::vector<float> dstArr(N);

    prefixSumRefImpl(srcArr, dstArr);
    REQUIRE(dstArr[1] == Catch::Approx(3.f));
    REQUIRE(dstArr[2] == Catch::Approx(6.f));
}

TEST_CASE("GLSL-PrefixSum", "") {
    vkc::initVulkan() | unwrap;

    constexpr float maxValidDiff = 0.001f;
    constexpr float maxValidAvgDiff = 0.00001f;

    constexpr int N = 256;

    // Src data
    std::vector<float> srcArr(N);
    std::mt19937 rdEngine;
    rdEngine.seed(37);
    std::uniform_real_distribution dist(0.0f, 1.0f);
    for (auto& val : srcArr) {
        val = dist(rdEngine);
    }

    // CPU Reference
    std::vector<float> dstArrCpuRef(N);
    prefixSumRefImpl(srcArr, dstArrCpuRef);

    std::vector<float> dstArrVk(N);

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
        vkc::StorageBufferBox::create(pDeviceBox, N * sizeof(float), vkc::StorageType::ReadOnly) | unwrap;
    vkc::StagingBufferBox srcArrStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, N * sizeof(float), vkc::StorageType::ReadOnly) | unwrap;
    const std::array srcMatBoxRefs{std::ref(srcArrBox)};
    vkc::StorageBufferBox dstArrBox =
        vkc::StorageBufferBox::create(pDeviceBox, N * sizeof(float), vkc::StorageType::ReadWrite) | unwrap;
    vkc::StagingBufferBox dstArrStagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, N * sizeof(float), vkc::StorageType::ReadWrite) | unwrap;
    const std::array dstArrBoxRefs{std::ref(dstArrBox)};
    const std::array dstStagingBufferRefs{std::ref(dstArrStagingBufferBox)};
    srcArrStagingBufferBox.upload((std::byte*)srcArr.data()) | unwrap;

    const std::vector descPoolSizes = genPoolSizes(srcArrBox, dstArrBox);
    vkc::DescPoolBox descPoolBox = vkc::DescPoolBox::create(pDeviceBox, descPoolSizes) | unwrap;

    const std::array presumDLayoutBindings = genDescSetLayoutBindings(srcArrBox, dstArrBox);
    vkc::DescSetLayoutBox presumDLayoutBox = vkc::DescSetLayoutBox::create(pDeviceBox, presumDLayoutBindings) | unwrap;
    const std::array presumDLayoutBoxCRefs{std::cref(presumDLayoutBox)};
    vkc::PipelineLayoutBox presumPLayoutBox =
        vkc::PipelineLayoutBox::create(pDeviceBox, presumDLayoutBoxCRefs) | unwrap;
    vkc::DescSetsBox presumDescSetsBox =
        vkc::DescSetsBox::create(pDeviceBox, descPoolBox, presumDLayoutBoxCRefs) | unwrap;
    const std::array presumWriteDescSets = genWriteDescSets(srcArrBox, dstArrBox);
    const std::array presumWriteDescSetss{std::span{presumWriteDescSets.begin(), presumWriteDescSets.end()}};
    presumDescSetsBox.updateDescSets(presumWriteDescSetss);

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox presumCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;

    SECTION("v0") {
        constexpr int groupSize = N;
        vkc::ShaderBox presumShaderBox = vkc::ShaderBox::create(pDeviceBox, shader::prefix_sum::v0::code) | unwrap;
        vkc::SpecConstantBox specConstantBox{groupSize, N};
        vkc::PipelineBox presumPipelineBox =
            vkc::PipelineBox::createCompute(pDeviceBox, presumPLayoutBox, presumShaderBox,
                                            specConstantBox.getSpecInfo()) |
            unwrap;

        presumCmdBufBox.begin() | unwrap;
        presumCmdBufBox.bindPipeline(presumPipelineBox);
        presumCmdBufBox.bindDescSets(presumDescSetsBox, presumPLayoutBox, vk::PipelineBindPoint::eCompute);
        presumCmdBufBox.recordPrepareReceive<vkc::StorageBufferBox>(srcMatBoxRefs);
        presumCmdBufBox.recordCopyStagingToBuffer(srcArrStagingBufferBox, srcArrBox);
        presumCmdBufBox.recordPrepareShaderRead<vkc::StorageBufferBox>(srcMatBoxRefs);
        presumCmdBufBox.recordPrepareShaderWrite(dstArrBoxRefs);
        presumCmdBufBox.recordDispatch(1, 1);
        presumCmdBufBox.recordPrepareSend(dstArrBoxRefs);
        presumCmdBufBox.recordCopyBufferToStaging(dstArrBox, dstArrStagingBufferBox);
        presumCmdBufBox.recordWaitDownloadComplete(dstStagingBufferRefs);
        presumCmdBufBox.end() | unwrap;

        queueBox.submit(presumCmdBufBox, fenceBox) | unwrap;
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
