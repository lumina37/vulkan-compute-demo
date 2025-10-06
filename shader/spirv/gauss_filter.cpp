#include <cstddef>
#include <span>

#include "spirv/gauss_filter.hpp"

namespace shader::gauss_filter {

namespace v0 {

namespace _detail {
#include "spirv/gauss_filter/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

namespace v1 {

namespace _detail {
#include "spirv/gauss_filter/v1.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v1

}  // namespace shader::gauss_filter
