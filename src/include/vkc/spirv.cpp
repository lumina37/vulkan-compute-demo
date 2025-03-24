#include <cstddef>
#include <span>

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/spirv.hpp"
#endif

namespace vkc {

namespace _spirv::gaussianBlur {

#include "vkc/_spirv/gaussianBlur.hlsl.h"

}

namespace _spirv::grayscale {

#include "vkc/_spirv/grayscale.hlsl.h"

}

const std::span<std::byte> gaussianBlurSpirvCode{(std::byte*)_spirv::gaussianBlur::g_main,
                                                 sizeof(_spirv::gaussianBlur::g_main)};
const std::span<std::byte> grayscaleSpirvCode{(std::byte*)_spirv::grayscale::g_main, sizeof(_spirv::grayscale::g_main)};

}  // namespace vkc
