#pragma once

#include <cstddef>
#include <span>

namespace shader::gaussFilter {

namespace v0 {

namespace _detail {
#include "spirv/gaussFilter/v0.h"
}

static const std::span code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

namespace v1 {

namespace _detail {
#include "spirv/gaussFilter/v1.h"
}

static const std::span code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v1

namespace rw {

namespace _detail {
#include "spirv/gaussFilter/rw.h"
}

static const std::span code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v1

}  // namespace shader::gaussFilter
