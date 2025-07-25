#pragma once

#include <expected>
#include <functional>
#include <print>

#include "vkc/device/instance.hpp"
#include "vkc/device/physical/box.hpp"
#include "vkc/device/physical/concepts.hpp"
#include "vkc/device/physical/props.hpp"
#include "vkc/device/score.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <CPhyDeviceProps TDProps_>
class PhyDeviceSet_ {
public:
    using TDProps = TDProps_;
    using TPhyDeviceWithProps = PhyDeviceWithProps_<TDProps>;
    using FnJudge = std::expected<float, Error> (*)(const TPhyDeviceWithProps&) noexcept;

private:
    PhyDeviceSet_(std::vector<TPhyDeviceWithProps>&& phyDevicesWithProps) noexcept;

public:
    PhyDeviceSet_(const PhyDeviceSet_&) = delete;
    PhyDeviceSet_(PhyDeviceSet_&&) noexcept = default;

    [[nodiscard]] static std::expected<PhyDeviceSet_, Error> create(const InstanceBox& instBox) noexcept;

    [[nodiscard]] std::expected<std::reference_wrapper<TPhyDeviceWithProps>, Error> select(
        const FnJudge& judge) noexcept;
    [[nodiscard]] std::expected<std::reference_wrapper<TPhyDeviceWithProps>, Error> selectDefault() noexcept;

private:
    std::vector<TPhyDeviceWithProps> phyDevicesWithProps_;
};

template <CPhyDeviceProps TProps>
PhyDeviceSet_<TProps>::PhyDeviceSet_(std::vector<TPhyDeviceWithProps>&& phyDevicesWithProps) noexcept
    : phyDevicesWithProps_(std::move(phyDevicesWithProps)) {}

template <CPhyDeviceProps TProps>
std::expected<PhyDeviceSet_<TProps>, Error> PhyDeviceSet_<TProps>::create(const InstanceBox& instBox) noexcept {
    const vk::Instance instance = instBox.getInstance();

    const auto [physicalDevicesRes, physicalDevices] = instance.enumeratePhysicalDevices();
    if (physicalDevicesRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, physicalDevicesRes}};
    }

    std::vector<TPhyDeviceWithProps> phyDevicesWithProps;
    phyDevicesWithProps.reserve(physicalDevices.size());
    for (const auto& physicalDevice : physicalDevices) {
        auto phyDeviceBoxRes = PhyDeviceBox::create(physicalDevice);
        if (!phyDeviceBoxRes) return std::unexpected{std::move(phyDeviceBoxRes.error())};
        auto& phyDeviceBox = phyDeviceBoxRes.value();

        auto phyDevicePropsRes = TDProps::create(phyDeviceBox);
        if (!phyDevicePropsRes) return std::unexpected{std::move(phyDevicePropsRes.error())};
        auto& phyDeviceProps = phyDevicePropsRes.value();

        phyDevicesWithProps.emplace_back(std::move(phyDeviceBox), std::move(phyDeviceProps));
    }

    return PhyDeviceSet_{std::move(phyDevicesWithProps)};
}

template <CPhyDeviceProps TProps>
auto PhyDeviceSet_<TProps>::select(const FnJudge& judge) noexcept
    -> std::expected<std::reference_wrapper<TPhyDeviceWithProps>, Error> {
    std::vector<Score<std::reference_wrapper<TPhyDeviceWithProps>>> scores;
    scores.reserve(phyDevicesWithProps_.size());

    const auto printDeviceInfo = [](const TPhyDeviceWithProps& deviceWithProps,
                                    const float score) -> std::expected<void, Error> {
        const vk::PhysicalDevice phyDevice = deviceWithProps.getPhyDeviceBox().getPhyDevice();
        const auto& phyDeviceProp = phyDevice.getProperties();
        const uint32_t apiVersion = deviceWithProps.getPhyDeviceProps().apiVersion;

        std::println("Candidate physical device: {}. Vk API version: {}.{}.{}. Score: {}",
                     phyDeviceProp.deviceName.data(), vk::apiVersionMajor(apiVersion), vk::apiVersionMinor(apiVersion),
                     vk::apiVersionPatch(apiVersion), score);

        return {};
    };

    for (auto& deviceWithProps : phyDevicesWithProps_) {
        auto scoreRes = judge(deviceWithProps);
        if (!scoreRes) return std::unexpected{std::move(scoreRes.error())};

        if constexpr (ENABLE_DEBUG) {
            auto printRes = printDeviceInfo(deviceWithProps, scoreRes.value());
            if (!printRes) return std::unexpected{std::move(printRes.error())};
        }

        scores.emplace_back(scoreRes.value(), std::ref(deviceWithProps));
    }

    if (scores.empty()) {
        return std::unexpected{Error{ECate::eVkC, ECode::eNoSupport, "no sufficient device"}};
    }

    auto maxScoreIt = std::max_element(scores.begin(), scores.end());
    return std::move(maxScoreIt->attachment);
}

template <CPhyDeviceProps TProps>
auto PhyDeviceSet_<TProps>::selectDefault() noexcept
    -> std::expected<std::reference_wrapper<TPhyDeviceWithProps>, Error> {
    constexpr auto defaultJudge = [](const TPhyDeviceWithProps& phyDeviceWithProps) noexcept {
        return phyDeviceWithProps.getPhyDeviceProps().score();
    };
    return select(defaultJudge);
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/set.cpp"
#endif
