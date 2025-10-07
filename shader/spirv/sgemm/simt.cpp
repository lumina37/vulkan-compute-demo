#include <cstddef>
#include <span>

#include "spirv/sgemm/simt.hpp"

namespace shader::sgemm::simt {

namespace v0 {

namespace _detail {
#include "spirv/sgemm/simt/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

namespace v1 {

namespace _detail {
#include "spirv/sgemm/simt/v1.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v1

namespace v2 {

namespace _detail {
#include "spirv/sgemm/simt/v2.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v2

namespace v3 {

namespace _detail {
#include "spirv/sgemm/simt/v3.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v3

namespace v4 {

namespace _detail {
#include "spirv/sgemm/simt/v4.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v4

namespace v5 {

namespace _detail {
#include "spirv/sgemm/simt/v5.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v5

}  // namespace shader::sgemm::simt
