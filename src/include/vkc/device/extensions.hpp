#pragma once

#include <algorithm>
#include <expected>
#include <functional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "vkc/device/concepts.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

template <CHasExtensionName TVkProps>
static constexpr std::string_view extractName(const TVkProps& props) {
    return props.extensionName;
}

template <CHasLayerName TVkProps>
static constexpr std::string_view extractName(const TVkProps& props) {
    return props.layerName;
}

template <CExt TExt_>
class ExtEntry_ {
public:
    using TExt = TExt_;

public:
    ExtEntry_(std::reference_wrapper<const TExt> extRef) noexcept;

    friend constexpr auto operator<=>(const ExtEntry_& lhs, const ExtEntry_& rhs) noexcept {
        return lhs.getKey() <=> rhs.getKey();
    }
    friend constexpr auto operator==(const ExtEntry_& lhs, const ExtEntry_& rhs) noexcept {
        return lhs.getKey() == rhs.getKey();
    }

    template <typename Self>
    [[nodiscard]] auto&& getKey(this Self&& self) noexcept {
        return std::forward_like<Self>(self).key_;
    }

private:
    std::string_view key_;
    std::reference_wrapper<const TExt> extRef_;
};

template <CExt TExt_>
class OrderedExtEntries_ {
public:
    using TExt = TExt_;
    using TEntry = ExtEntry_<TExt>;

private:
    OrderedExtEntries_(std::vector<TExt>&& exts, std::vector<TEntry>&& extEntries) noexcept;

public:
    [[nodiscard]] static std::expected<OrderedExtEntries_, Error> create(std::vector<TExt>&& exts) noexcept;

    [[nodiscard]] bool has(std::string_view query) const noexcept;

private:
    std::vector<TExt> exts_;
    std::vector<TEntry> extEntries_;
};

namespace rgs = std::ranges;

template <CExt TExt_>
ExtEntry_<TExt_>::ExtEntry_(std::reference_wrapper<const TExt> extRef) noexcept
    : key_(extractName(extRef.get())), extRef_(extRef) {}

template <CExt TExt_>
OrderedExtEntries_<TExt_>::OrderedExtEntries_(std::vector<TExt>&& exts, std::vector<TEntry>&& extEntries) noexcept
    : exts_(std::move(exts)), extEntries_(std::move(extEntries)) {}

template <CExt TExt_>
std::expected<OrderedExtEntries_<TExt_>, Error> OrderedExtEntries_<TExt_>::create(std::vector<TExt>&& exts) noexcept {
    const auto toEntry = [](const TExt& ext) { return TEntry{std::cref(ext)}; };

    auto extEntries = exts | rgs::views::transform(toEntry) | rgs::to<std::vector>();
    rgs::sort(extEntries);

    return OrderedExtEntries_{std::move(exts), std::move(extEntries)};
}

template <CExt TExt>
bool OrderedExtEntries_<TExt>::has(std::string_view query) const noexcept {
    return rgs::binary_search(extEntries_, query, std::less{}, [](const TEntry& entry) { return entry.getKey(); });
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/extensions.cpp"
#endif
