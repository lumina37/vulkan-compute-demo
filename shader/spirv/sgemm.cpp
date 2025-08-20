#include <cstddef>
#include <span>

#include "spirv/sgemm.hpp"

namespace shader::sgemm {

namespace v0 {

namespace _detail {
#include "spirv/sgemm/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

namespace v1 {

namespace _detail {
#include "spirv/sgemm/v1.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v1

namespace v2 {

namespace _detail {
#include "spirv/sgemm/v2.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v2

namespace v3 {

namespace _detail {
#include "spirv/sgemm/v3.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v3

}  // namespace shader::sgemm
