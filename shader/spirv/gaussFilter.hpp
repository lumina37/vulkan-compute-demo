#pragma once

#include <cstddef>
#include <span>

namespace shader {

namespace _spirv::gaussFilterV0 {

#include "spirv/gaussFilterV0.h"

}

namespace _spirv::gaussFilterV1 {

#include "spirv/gaussFilterV1.hlsl.h"

}

namespace _spirv::gaussFilterV2 {

#include "spirv/gaussFilterV2.hlsl.h"

}


static const std::span gaussFilterV0SpirvCode{(std::byte*)_spirv::gaussFilterV0::spirvCode,
                                              sizeof(_spirv::gaussFilterV0::spirvCode)};
static const std::span gaussFilterV1SpirvCode{(std::byte*)_spirv::gaussFilterV1::g_main,
                                              sizeof(_spirv::gaussFilterV1::g_main)};
static const std::span gaussFilterV2SpirvCode{(std::byte*)_spirv::gaussFilterV2::g_main,
                                              sizeof(_spirv::gaussFilterV2::g_main)};

}  // namespace shader
