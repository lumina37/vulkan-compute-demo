#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <span>

#include "vkc.hpp"
#include "vkc_bin_helper.hpp"

int main() {
    vkc::StbImageManager srcImage = vkc::StbImageManager::createFromPath("in.png") | unwrap;

    // Device
    vkc::DefaultInstanceProps instProps = vkc::DefaultInstanceProps::create() | unwrap;
    if (!instProps.layers.has("VK_LAYER_KHRONOS_validation")) {
        std::println(std::cerr, "VK_LAYER_KHRONOS_validation not supported");
        return -1;
    }

    vkc::WindowManager::globalInit() | unwrap;  // only call once
    auto instExtNames = vkc::WindowManager::getExtensions() | unwrap;
    constexpr std::string_view validationLayerName{"VK_LAYER_KHRONOS_validation"};
    constexpr std::array instLayerNames{validationLayerName};
    auto pInstMgr = std::make_shared<vkc::InstanceManager>(
        vkc::InstanceManager::createWithExts(instExtNames, instLayerNames) | unwrap);
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(*pInstMgr) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
    vkc::PhyDeviceManager& phyDeviceMgr = phyDeviceWithProps.getPhyDeviceMgr();
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceMgr) | unwrap;
    constexpr std::string_view swapchainExtName{vk::KHRSwapchainExtensionName};
    constexpr std::array deviceExtNames{swapchainExtName};
    auto pDeviceMgr = std::make_shared<vkc::DeviceManager>(
        vkc::DeviceManager::createWithExts(phyDeviceMgr, {vk::QueueFlagBits::eCompute, computeQFamilyIdx},
                                           deviceExtNames) |
        unwrap);
    vkc::QueueManager queueMgr = vkc::QueueManager::create(*pDeviceMgr, vk::QueueFlagBits::eCompute) | unwrap;

    // Swapchain
    vkc::WindowManager windowMgr = vkc::WindowManager::create(srcImage.getExtent().extent()) | unwrap;
    vkc::SurfaceManager surfaceMgr = vkc::SurfaceManager::create(pInstMgr, windowMgr) | unwrap;
    const std::array familyIndices{computeQFamilyIdx};
    vkc::SwapChainManager swapChainMgr =
        vkc::SwapChainManager::create(phyDeviceMgr, pDeviceMgr, surfaceMgr, familyIndices, srcImage.getExtent()) |
        unwrap;

    // Command Buffer
    vkc::FenceManager fenceMgr = vkc::FenceManager::create(pDeviceMgr) | unwrap;
    vkc::SemaphoreManager semaphoreMgr = vkc::SemaphoreManager::create(pDeviceMgr) | unwrap;
    auto pCommandPoolMgr = std::make_shared<vkc::CommandPoolManager>(
        vkc::CommandPoolManager::create(pDeviceMgr, computeQFamilyIdx) | unwrap);
    vkc::CommandBufferManager presentCmdBufMgr =
        vkc::CommandBufferManager::create(pDeviceMgr, pCommandPoolMgr) | unwrap;
    vkc::TimestampQueryPoolManager queryPoolMgr =
        vkc::TimestampQueryPoolManager::create(pDeviceMgr, 6, phyDeviceWithProps.getPhyDeviceProps().timestampPeriod) |
        unwrap;

    // Main Loop
    while (!glfwWindowShouldClose(windowMgr.getWindow())) {
        Timer loopTimer;
        loopTimer.begin();

        const uint32_t imageIndex = swapChainMgr.acquireImageIndex(semaphoreMgr) | unwrap;

        loopTimer.end();
        std::println("Acquire image timecost: {} ms", loopTimer.durationMs());
        loopTimer.begin();

        vkc::PresentImageManager& presentImageMgr = swapChainMgr.getImageMgr((int)imageIndex);
        presentImageMgr.upload(srcImage.getPData()) | unwrap;
        const std::array presentImageMgrRefs{std::ref(presentImageMgr)};

        presentCmdBufMgr.begin() | unwrap;
        presentCmdBufMgr.recordPrepareReceiveBeforeDispatch<vkc::PresentImageManager>(presentImageMgrRefs);
        presentCmdBufMgr.recordCopyStagingToSrc(presentImageMgr);
        presentCmdBufMgr.recordPreparePresent(presentImageMgrRefs);
        presentCmdBufMgr.end() | unwrap;

        presentCmdBufMgr.submitAndWaitPreTask(queueMgr, semaphoreMgr, vk::PipelineStageFlagBits::eTransfer, fenceMgr) |
            unwrap;
        fenceMgr.wait() | unwrap;
        fenceMgr.reset() | unwrap;

        swapChainMgr.present(queueMgr, imageIndex) | unwrap;

        loopTimer.end();
        std::println("Present timecost: {} ms", loopTimer.durationMs());

        glfwPollEvents();
    }

    vkc::WindowManager::globalDestroy();
}
