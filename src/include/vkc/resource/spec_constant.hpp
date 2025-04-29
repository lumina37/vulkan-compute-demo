#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <vulkan/vulkan.hpp>

namespace vkc {

template <typename... TSc>
    requires(std::is_trivially_copyable_v<TSc> && ...)
class SpecConstantManager {
public:
    constexpr SpecConstantManager(TSc... specConstants) noexcept;

    template <typename Self>
    [[nodiscard]] constexpr auto&& getSpecInfo(this Self&& self) noexcept {
        return std::forward_like<Self>(self).specInfo_;
    }

private:
    std::tuple<TSc...> specConstants_;

    std::array<vk::SpecializationMapEntry, sizeof...(TSc)> specMapEntries_;
    vk::SpecializationInfo specInfo_;
};
template <typename... TSc>
    requires(std::is_trivially_copyable_v<TSc> && ...)
constexpr SpecConstantManager<TSc...>::SpecConstantManager(TSc... specConstants) noexcept
    : specConstants_(specConstants...) {
    const auto genSpecMapEntry = [&]<size_t index>() {
        const auto& specConstant = std::get<index>(specConstants_);
        const size_t offset = (size_t)&specConstant - (size_t)&specConstants_;

        vk::SpecializationMapEntry specMapEntry;
        specMapEntry.setConstantID((uint32_t)index);
        specMapEntry.setOffset((uint32_t)offset);
        specMapEntry.setSize(sizeof(specConstant));

        return specMapEntry;
    };

    const auto genSpecMapEnrtyHelper = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array { genSpecMapEntry.template operator()<Is>()... };
    };

    specMapEntries_ = genSpecMapEnrtyHelper(std::index_sequence_for<TSc...>{});

    specInfo_.setMapEntries(specMapEntries_);
    specInfo_.setDataSize(sizeof(specConstants_));
    specInfo_.setPData(&specConstants_);
}

}  // namespace vkc
