#include <cstddef>
#include <span>

#include "spirv/prefix_sum.hpp"

namespace shader::prefix_sum {

namespace v0 {

namespace _detail {
#include "spirv/prefix_sum/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

}  // namespace shader::prefix_sum
