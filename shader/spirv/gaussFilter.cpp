#include <cstddef>
#include <span>

#include "spirv/gaussFilter.hpp"

namespace shader::gaussFilter {

namespace v0 {

namespace _detail {
#include "spirv/gaussFilter/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

namespace v1 {

namespace _detail {
#include "spirv/gaussFilter/v1.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v1

}  // namespace shader::gaussFilter
