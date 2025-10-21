// types.glsl

#if !defined(GGML_TYPES_COMP)
#define GGML_TYPES_COMP

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require

#if defined(DATA_A_F32)
#define QUANT_K 1
#define QUANT_R 1

#if LOAD_VEC_A == 4
#define A_TYPE vec4
#elif LOAD_VEC_A == 8
#define A_TYPE mat2x4
#else
#define A_TYPE float
#endif
#endif

#if defined(DATA_A_F16)
#define QUANT_K 1
#define QUANT_R 1

#if LOAD_VEC_A == 4
#define A_TYPE f16vec4
#elif LOAD_VEC_A == 8
#define A_TYPE f16mat2x4
#else
#define A_TYPE float16_t
#endif
#endif

#endif // !defined(GGML_TYPES_COMP)