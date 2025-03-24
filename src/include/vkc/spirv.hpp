#pragma once

#include <cstddef>
#include <span>

namespace vkc {

extern const std::span<std::byte> gaussianBlurSpirvCode;
extern const std::span<std::byte> grayscaleSpirvCode;

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/spriv.cpp"
#endif
