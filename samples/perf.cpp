#include <array>
#include <iostream>
#include <memory>
#include <print>
#include <span>

#include "vkc.hpp"
#include "vkc_bin_helper.hpp"

int main() {
    vkc::initVulkan() | unwrap;

    // Device
    vkc::DefaultInstanceProps instProps = vkc::DefaultInstanceProps::create() | unwrap;
    if (!instProps.layers.has("VK_LAYER_KHRONOS_validation")) {
        std::println(std::cerr, "VK_LAYER_KHRONOS_validation not supported");
        return -1;
    }
    vkc::InstanceBox instBox = vkc::InstanceBox::create() | unwrap;
    vkc::PhyDeviceSet phyDeviceSet = vkc::PhyDeviceSet::create(instBox) | unwrap;
    vkc::PhyDeviceWithProps& phyDeviceWithProps = (phyDeviceSet.selectDefault() | unwrap).get();
    vkc::PhyDeviceBox& phyDeviceBox = phyDeviceWithProps.getPhyDeviceBox();
    const uint32_t computeQFamilyIdx = defaultComputeQFamilyIndex(phyDeviceBox) | unwrap;
    vkc::PerfCounterProps perfProps = vkc::PerfCounterProps::create(phyDeviceBox, computeQFamilyIdx) | unwrap;

    for (auto& prop : perfProps.perfCounters) {
        std::println("--------------------");
        std::println("name: {}", prop.getName());
        std::println("cate: {}", prop.getCategory());
        std::println("desc: {}", prop.getDescription());
    }

    constexpr std::string_view perfQueryExtName{vk::KHRPerformanceQueryExtensionName};
    constexpr std::array deviceExtNames{perfQueryExtName};
    auto pDeviceBox = std::make_shared<vkc::DeviceBox>(
        vkc::DeviceBox::createWithExts(phyDeviceBox, {vk::QueueFlagBits::eCompute, computeQFamilyIdx}, deviceExtNames) |
        unwrap);
}
