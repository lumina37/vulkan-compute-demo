#pragma once

#include <cstddef>
#include <span>

namespace shader::grayscale {

namespace ro {

namespace _detail {
#include "spirv/grayscale/ro.h"
}

static const std::span code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace ro

namespace rw {

namespace _detail {
#include "spirv/grayscale/rw.h"
}

static const std::span code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace rw

}  // namespace shader::grayscale
