#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <span>

#include "vkc.hpp"
#include "vkc_helper.hpp"

int main() {
    vkc::initVulkan() | unwrap;
    vkc::initGLFW() | unwrap;

    vkc::StbImageBox srcImage = vkc::StbImageBox::createFromPath("in.png") | unwrap;

    // Device
    vkc::DefaultInstanceProps instProps = vkc::DefaultInstanceProps::create() | unwrap;
    if (!instProps.layers.has("VK_LAYER_KHRONOS_validation")) {
        std::println(std::cerr, "VK_LAYER_KHRONOS_validation not supported");
        return -1;
    }

    auto instExtNames = vkc::WindowBox::getExtensions() | unwrap;
    constexpr std::string_view validationLayerName{"VK_LAYER_KHRONOS_validation"};
    constexpr std::array instLayerNames{validationLayerName};
    auto pInstBox =
        std::make_shared<vkc::InstanceBox>(vkc::InstanceBox::createWithExts(instExtNames, instLayerNames) | unwrap);
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(*pInstBox) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    constexpr std::string_view swapchainExtName{vk::KHRSwapchainExtensionName};
    constexpr std::array deviceExtNames{swapchainExtName};
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::createWithExts(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}, deviceExtNames,
                                       nullptr) |
        unwrap);
    vkc::QueueBox queueBox = vkc::QueueBox::create(*pDeviceBox, vk::QueueFlagBits::eCompute) | unwrap;

    // Swapchain
    vkc::WindowBox windowBox = vkc::WindowBox::create(srcImage.getExtent().extent()) | unwrap;
    vkc::SurfaceBox surfaceBox = vkc::SurfaceBox::create(pInstBox, windowBox) | unwrap;
    const std::array familyIndices{computeQFamilyIdx};
    vkc::SwapchainBox swapChainBox =
        vkc::SwapchainBox::create(pDeviceBox, surfaceBox, familyIndices, srcImage.getExtent()) | unwrap;
    vkc::StagingBufferBox stagingBufferBox =
        vkc::StagingBufferBox::create(pDeviceBox, srcImage.getExtent().size(), vkc::StorageType::ReadOnly) | unwrap;
    stagingBufferBox.upload(srcImage.getPData()) | unwrap;

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    vkc::SemaphoreBox semaphoreBox = vkc::SemaphoreBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox presentCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;

    // Main Loop
    while (!glfwWindowShouldClose(windowBox.getWindow())) {
        Timer loopTimer;

        loopTimer.begin();
        const uint32_t imageIndex = swapChainBox.acquireImageIndex(semaphoreBox) | unwrap;
        loopTimer.end();

        std::println("Acquire image timecost: {} ms", loopTimer.durationMs());

        vkc::PresentImageBox& presentImageBox = swapChainBox.getPresentImageBox(imageIndex);
        const std::array presentImageBoxRefs{std::ref(presentImageBox)};

        presentCmdBufBox.begin() | unwrap;
        presentCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::PresentImageBox>(presentImageBoxRefs);
        presentCmdBufBox.recordCopyStagingToSrc(stagingBufferBox, presentImageBox);
        presentCmdBufBox.recordPreparePresent(presentImageBoxRefs);
        presentCmdBufBox.end() | unwrap;

        queueBox.submitAndWaitSemaphore(presentCmdBufBox, semaphoreBox, vk::PipelineStageFlagBits::eTransfer,
                                        fenceBox) |
            unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        loopTimer.begin();
        queueBox.present(swapChainBox, imageIndex) | unwrap;
        loopTimer.end();

        std::println("Present timecost: {} ms", loopTimer.durationMs());

        glfwPollEvents();
    }

    vkc::WindowBox::globalDestroy();
}
