#pragma once

#include <cstddef>
#include <span>

namespace shader {

namespace _spirv::gaussFilterImgload {

#include "spirv/gaussFilterImgload.hlsl.h"

}

static const std::span gaussFilterImgloadSpirvCode{(std::byte*)_spirv::gaussFilterImgload::g_main,
                                             sizeof(_spirv::gaussFilterImgload::g_main)};

}  // namespace vkc
