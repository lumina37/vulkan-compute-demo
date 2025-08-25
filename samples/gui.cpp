#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <span>

#include "vkc.hpp"
#include "vkc_helper.hpp"

int main() {
    vkc::initVulkan() | unwrap;

    vkc::StbImageBox srcImage = vkc::StbImageBox::createFromPath("in.png") | unwrap;

    // Device
    vkc::DefaultInstanceProps instProps = vkc::DefaultInstanceProps::create() | unwrap;
    if (!instProps.layers.has("VK_LAYER_KHRONOS_validation")) {
        std::println(std::cerr, "VK_LAYER_KHRONOS_validation not supported");
        return -1;
    }

    vkc::WindowBox::globalInit() | unwrap;  // only call once
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
        vkc::SwapchainBox::create(phyDeviceBox, pDeviceBox, surfaceBox, familyIndices, srcImage.getExtent()) | unwrap;

    // Command Buffer
    vkc::FenceBox fenceBox = vkc::FenceBox::create(pDeviceBox) | unwrap;
    vkc::SemaphoreBox semaphoreBox = vkc::SemaphoreBox::create(pDeviceBox) | unwrap;
    auto pCommandPoolBox =
        std::make_shared<vkc::CommandPoolBox>(vkc::CommandPoolBox::create(pDeviceBox, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferBox presentCmdBufBox = vkc::CommandBufferBox::create(pDeviceBox, pCommandPoolBox) | unwrap;
    vkc::TimestampQueryPoolBox queryPoolBox =
        vkc::TimestampQueryPoolBox::create(pDeviceBox, 6, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
        unwrap;

    // Main Loop
    while (!glfwWindowShouldClose(windowBox.getWindow())) {
        Timer loopTimer;
        loopTimer.begin();

        const uint32_t imageIndex = swapChainBox.acquireImageIndex(semaphoreBox) | unwrap;

        loopTimer.end();
        std::println("Acquire image timecost: {} ms", loopTimer.durationMs());
        loopTimer.begin();

        vkc::PresentImageBox& presentImageBox = swapChainBox.getImageBox((int)imageIndex);
        presentImageBox.upload(srcImage.getPData()) | unwrap;
        const std::array presentImageBoxRefs{std::ref(presentImageBox)};

        presentCmdBufBox.begin() | unwrap;
        presentCmdBufBox.recordPrepareReceiveBeforeDispatch<vkc::PresentImageBox>(presentImageBoxRefs);
        presentCmdBufBox.recordCopyStagingToSrc(presentImageBox);
        presentCmdBufBox.recordPreparePresent(presentImageBoxRefs);
        presentCmdBufBox.end() | unwrap;

        queueBox.submitAndWaitSemaphore(presentCmdBufBox, semaphoreBox, vk::PipelineStageFlagBits::eTransfer,
                                        fenceBox) |
            unwrap;
        fenceBox.wait() | unwrap;
        fenceBox.reset() | unwrap;

        queueBox.present(swapChainBox, imageIndex) | unwrap;

        loopTimer.end();
        std::println("Present timecost: {} ms", loopTimer.durationMs());

        glfwPollEvents();
    }

    vkc::WindowBox::globalDestroy();
}
