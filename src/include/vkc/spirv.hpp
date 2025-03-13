#pragma once

namespace vkc {

namespace _spirv::gaussianBlur {

#include "vkc/_spirv/gaussianBlur.hlsl.h"

}

const std::span<std::byte> gaussianBlurSpirvCode{(std::byte*)_spirv::gaussianBlur::g_main,
                                                 sizeof(_spirv::gaussianBlur::g_main)};

}  // namespace vkc
