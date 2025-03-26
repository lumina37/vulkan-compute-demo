#if 0
; SPIR-V
; Version: 1.0
; Generator: Google spiregg; 0
; Bound: 122
; Schema: 0
               OpCapability Shader
               OpCapability ImageQuery
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
               OpExecutionMode %main LocalSize 16 16 1
               OpSource HLSL 600
               OpName %type_ConstantBuffer_PushConstants "type.ConstantBuffer.PushConstants"
               OpMemberName %type_ConstantBuffer_PushConstants 0 "kernelSize"
               OpName %pc "pc"
               OpName %type_2d_image "type.2d.image"
               OpName %srcTex "srcTex"
               OpName %type_sampler "type.sampler"
               OpName %srcSampler "srcSampler"
               OpName %type_2d_image_0 "type.2d.image"
               OpName %dstImage "dstImage"
               OpName %type_ConstantBuffer_UBO "type.ConstantBuffer.UBO"
               OpMemberName %type_ConstantBuffer_UBO 0 "weights"
               OpName %ubo "ubo"
               OpName %main "main"
               OpName %type_sampled_image "type.sampled.image"
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %srcTex DescriptorSet 0
               OpDecorate %srcTex Binding 0
               OpDecorate %srcSampler DescriptorSet 0
               OpDecorate %srcSampler Binding 1
               OpDecorate %dstImage DescriptorSet 0
               OpDecorate %dstImage Binding 2
               OpDecorate %ubo DescriptorSet 0
               OpDecorate %ubo Binding 3
               OpMemberDecorate %type_ConstantBuffer_PushConstants 0 Offset 0
               OpDecorate %type_ConstantBuffer_PushConstants Block
               OpDecorate %_arr_v4float_uint_4 ArrayStride 16
               OpMemberDecorate %type_ConstantBuffer_UBO 0 Offset 0
               OpDecorate %type_ConstantBuffer_UBO Block
               OpDecorate %16 NoContraction
               OpDecorate %17 NoContraction
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %int_2 = OpConstant %int 2
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %v4float = OpTypeVector %float 4
         %27 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
  %float_0_5 = OpConstant %float 0.5
    %v2float = OpTypeVector %float 2
         %30 = OpConstantComposite %v2float %float_0_5 %float_0_5
      %int_3 = OpConstant %int 3
    %float_1 = OpConstant %float 1
%type_ConstantBuffer_PushConstants = OpTypeStruct %int
%_ptr_PushConstant_type_ConstantBuffer_PushConstants = OpTypePointer PushConstant %type_ConstantBuffer_PushConstants
%type_2d_image = OpTypeImage %float 2D 2 0 0 1 Unknown
%_ptr_UniformConstant_type_2d_image = OpTypePointer UniformConstant %type_2d_image
%type_sampler = OpTypeSampler
%_ptr_UniformConstant_type_sampler = OpTypePointer UniformConstant %type_sampler
%type_2d_image_0 = OpTypeImage %float 2D 2 0 0 2 Rgba8
%_ptr_UniformConstant_type_2d_image_0 = OpTypePointer UniformConstant %type_2d_image_0
       %uint = OpTypeInt 32 0
     %uint_4 = OpConstant %uint 4
%_arr_v4float_uint_4 = OpTypeArray %v4float %uint_4
%type_ConstantBuffer_UBO = OpTypeStruct %_arr_v4float_uint_4
%_ptr_Uniform_type_ConstantBuffer_UBO = OpTypePointer Uniform %type_ConstantBuffer_UBO
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %43 = OpTypeFunction %void
      %v2int = OpTypeVector %int 2
     %v2uint = OpTypeVector %uint 2
%_ptr_PushConstant_int = OpTypePointer PushConstant %int
%type_sampled_image = OpTypeSampledImage %type_2d_image
%_ptr_Uniform_float = OpTypePointer Uniform %float
         %pc = OpVariable %_ptr_PushConstant_type_ConstantBuffer_PushConstants PushConstant
     %srcTex = OpVariable %_ptr_UniformConstant_type_2d_image UniformConstant
 %srcSampler = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant
   %dstImage = OpVariable %_ptr_UniformConstant_type_2d_image_0 UniformConstant
        %ubo = OpVariable %_ptr_Uniform_type_ConstantBuffer_UBO Uniform
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
     %uint_0 = OpConstant %uint 0
     %int_n2 = OpConstant %int -2
       %main = OpFunction %void None %43
         %50 = OpLabel
         %51 = OpLoad %v3uint %gl_GlobalInvocationID
               OpSelectionMerge %52 None
               OpSwitch %uint_0 %53
         %53 = OpLabel
         %54 = OpVectorShuffle %v2uint %51 %51 0 1
         %55 = OpBitcast %v2int %54
         %56 = OpLoad %type_2d_image_0 %dstImage
         %57 = OpImageQuerySize %v2uint %56
         %58 = OpCompositeExtract %uint %57 0
         %59 = OpBitcast %int %58
         %60 = OpCompositeExtract %uint %57 1
         %61 = OpBitcast %int %60
         %62 = OpCompositeConstruct %v2int %59 %61
         %63 = OpCompositeExtract %int %55 0
         %64 = OpSGreaterThanEqual %bool %63 %59
         %65 = OpLogicalNot %bool %64
               OpSelectionMerge %66 None
               OpBranchConditional %65 %67 %66
         %67 = OpLabel
         %68 = OpCompositeExtract %int %55 1
         %69 = OpSGreaterThanEqual %bool %68 %61
               OpBranch %66
         %66 = OpLabel
         %70 = OpPhi %bool %true %53 %69 %67
               OpSelectionMerge %71 None
               OpBranchConditional %70 %72 %71
         %72 = OpLabel
               OpBranch %52
         %71 = OpLabel
         %73 = OpAccessChain %_ptr_PushConstant_int %pc %int_0
         %74 = OpLoad %int %73
         %75 = OpSDiv %int %74 %int_2
         %76 = OpSDiv %int %74 %int_n2
               OpBranch %77
         %77 = OpLabel
         %78 = OpPhi %v4float %27 %71 %17 %79
         %80 = OpPhi %int %76 %71 %81 %79
         %82 = OpSLessThanEqual %bool %80 %75
               OpLoopMerge %83 %79 None
               OpBranchConditional %82 %84 %83
         %84 = OpLabel
               OpBranch %85
         %85 = OpLabel
         %86 = OpPhi %v4float %27 %84 %16 %87
         %88 = OpPhi %int %76 %84 %89 %87
         %90 = OpSLessThanEqual %bool %88 %75
               OpLoopMerge %91 %87 None
               OpBranchConditional %90 %87 %91
         %87 = OpLabel
         %92 = OpCompositeConstruct %v2int %88 %80
         %93 = OpIAdd %v2int %55 %92
         %94 = OpConvertSToF %v2float %93
         %95 = OpFAdd %v2float %94 %30
         %96 = OpConvertSToF %v2float %62
         %97 = OpFDiv %v2float %95 %96
         %98 = OpLoad %type_2d_image %srcTex
         %99 = OpLoad %type_sampler %srcSampler
        %100 = OpSampledImage %type_sampled_image %98 %99
        %101 = OpImageSampleExplicitLod %v4float %100 %97 Lod %float_0
        %102 = OpExtInst %int %1 SAbs %88
        %103 = OpShiftRightArithmetic %int %102 %int_2
        %104 = OpBitwiseAnd %int %102 %int_3
        %105 = OpBitcast %uint %104
        %106 = OpAccessChain %_ptr_Uniform_float %ubo %int_0 %103 %105
        %107 = OpLoad %float %106
        %108 = OpCompositeConstruct %v4float %107 %107 %107 %107
         %16 = OpExtInst %v4float %1 Fma %101 %108 %86
         %89 = OpIAdd %int %88 %int_1
               OpBranch %85
         %91 = OpLabel
        %109 = OpExtInst %int %1 SAbs %80
        %110 = OpShiftRightArithmetic %int %109 %int_2
        %111 = OpBitwiseAnd %int %109 %int_3
        %112 = OpBitcast %uint %111
        %113 = OpAccessChain %_ptr_Uniform_float %ubo %int_0 %110 %112
        %114 = OpLoad %float %113
        %115 = OpCompositeConstruct %v4float %114 %114 %114 %114
         %17 = OpExtInst %v4float %1 Fma %86 %115 %78
               OpBranch %79
         %79 = OpLabel
         %81 = OpIAdd %int %80 %int_1
               OpBranch %77
         %83 = OpLabel
        %116 = OpCompositeExtract %float %78 0
        %117 = OpCompositeExtract %float %78 1
        %118 = OpCompositeExtract %float %78 2
        %119 = OpCompositeConstruct %v4float %116 %117 %118 %float_1
        %120 = OpBitcast %v2uint %55
        %121 = OpLoad %type_2d_image_0 %dstImage
               OpImageWrite %121 %120 %119 None
               OpBranch %52
         %52 = OpLabel
               OpReturn
               OpFunctionEnd

#endif

const unsigned char g_main[] = {
  0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x7a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x06, 0x00, 0x01, 0x00, 0x00, 0x00, 0x47, 0x4c, 0x53, 0x4c,
  0x2e, 0x73, 0x74, 0x64, 0x2e, 0x34, 0x35, 0x30, 0x00, 0x00, 0x00, 0x00,
  0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x0f, 0x00, 0x06, 0x00, 0x05, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x06, 0x00, 0x02, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x03, 0x00, 0x05, 0x00, 0x00, 0x00, 0x58, 0x02, 0x00, 0x00,
  0x05, 0x00, 0x0b, 0x00, 0x04, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65,
  0x2e, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x42, 0x75, 0x66,
  0x66, 0x65, 0x72, 0x2e, 0x50, 0x75, 0x73, 0x68, 0x43, 0x6f, 0x6e, 0x73,
  0x74, 0x61, 0x6e, 0x74, 0x73, 0x00, 0x00, 0x00, 0x06, 0x00, 0x06, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6b, 0x65, 0x72, 0x6e,
  0x65, 0x6c, 0x53, 0x69, 0x7a, 0x65, 0x00, 0x00, 0x05, 0x00, 0x03, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x70, 0x63, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65, 0x2e, 0x32, 0x64, 0x2e,
  0x69, 0x6d, 0x61, 0x67, 0x65, 0x00, 0x00, 0x00, 0x05, 0x00, 0x04, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x73, 0x72, 0x63, 0x54, 0x65, 0x78, 0x00, 0x00,
  0x05, 0x00, 0x06, 0x00, 0x08, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65,
  0x2e, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x00, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x05, 0x00, 0x09, 0x00, 0x00, 0x00, 0x73, 0x72, 0x63, 0x53,
  0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65, 0x2e, 0x32, 0x64, 0x2e,
  0x69, 0x6d, 0x61, 0x67, 0x65, 0x00, 0x00, 0x00, 0x05, 0x00, 0x05, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x64, 0x73, 0x74, 0x49, 0x6d, 0x61, 0x67, 0x65,
  0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x74, 0x79, 0x70, 0x65, 0x2e, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e,
  0x74, 0x42, 0x75, 0x66, 0x66, 0x65, 0x72, 0x2e, 0x55, 0x42, 0x4f, 0x00,
  0x06, 0x00, 0x05, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73, 0x00, 0x05, 0x00, 0x03, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x75, 0x62, 0x6f, 0x00, 0x05, 0x00, 0x04, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x07, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65,
  0x2e, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x64, 0x2e, 0x69, 0x6d, 0x61,
  0x67, 0x65, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x03, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x03, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x03, 0x00, 0x10, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x03, 0x00, 0x11, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x02, 0x00, 0x15, 0x00, 0x00, 0x00, 0x29, 0x00, 0x03, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x16, 0x00, 0x03, 0x00, 0x18, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x07, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
  0x19, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x3f, 0x17, 0x00, 0x04, 0x00, 0x1d, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x05, 0x00,
  0x1d, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x1f, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
  0x1e, 0x00, 0x03, 0x00, 0x04, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x21, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x19, 0x00, 0x09, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x22, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x02, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x19, 0x00, 0x09, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
  0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x04, 0x00, 0x25, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x25, 0x00, 0x00, 0x00,
  0x26, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x04, 0x00,
  0x0f, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00,
  0x1e, 0x00, 0x03, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x27, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x28, 0x00, 0x00, 0x00,
  0x25, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
  0x29, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x02, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x21, 0x00, 0x03, 0x00,
  0x2b, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x04, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x2e, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x03, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
  0x2f, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x3b, 0x00, 0x04, 0x00, 0x21, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x22, 0x00, 0x00, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
  0x23, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x3b, 0x00, 0x04, 0x00, 0x24, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x27, 0x00, 0x00, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
  0x29, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x25, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x31, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0x36, 0x00, 0x05, 0x00,
  0x2a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x28, 0x00, 0x00, 0x00, 0x33, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xfb, 0x00, 0x03, 0x00, 0x30, 0x00, 0x00, 0x00,
  0x35, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x35, 0x00, 0x00, 0x00,
  0x4f, 0x00, 0x07, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
  0x33, 0x00, 0x00, 0x00, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00, 0x2c, 0x00, 0x00, 0x00,
  0x37, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x68, 0x00, 0x04, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x39, 0x00, 0x00, 0x00,
  0x38, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x25, 0x00, 0x00, 0x00,
  0x3a, 0x00, 0x00, 0x00, 0x39, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x00,
  0x3a, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x25, 0x00, 0x00, 0x00,
  0x3c, 0x00, 0x00, 0x00, 0x39, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00,
  0x3c, 0x00, 0x00, 0x00, 0x50, 0x00, 0x05, 0x00, 0x2c, 0x00, 0x00, 0x00,
  0x3e, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00,
  0x37, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xaf, 0x00, 0x05, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00,
  0x3b, 0x00, 0x00, 0x00, 0xa8, 0x00, 0x04, 0x00, 0x15, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00,
  0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00,
  0x41, 0x00, 0x00, 0x00, 0x43, 0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x43, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x37, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xaf, 0x00, 0x05, 0x00, 0x15, 0x00, 0x00, 0x00,
  0x45, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x42, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x42, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00, 0x15, 0x00, 0x00, 0x00,
  0x46, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00,
  0x45, 0x00, 0x00, 0x00, 0x43, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00,
  0x47, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00,
  0x46, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x48, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x34, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x47, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x05, 0x00, 0x2e, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00,
  0x87, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00,
  0x4a, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x87, 0x00, 0x05, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x4c, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00,
  0x31, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00, 0x4d, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x4d, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x4e, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x00, 0x00,
  0xf5, 0x00, 0x07, 0x00, 0x12, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x4c, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00,
  0x4f, 0x00, 0x00, 0x00, 0xb3, 0x00, 0x05, 0x00, 0x15, 0x00, 0x00, 0x00,
  0x52, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00,
  0xf6, 0x00, 0x04, 0x00, 0x53, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00, 0x52, 0x00, 0x00, 0x00,
  0x54, 0x00, 0x00, 0x00, 0x53, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x54, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00, 0x55, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x55, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x56, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00,
  0x54, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x57, 0x00, 0x00, 0x00,
  0xf5, 0x00, 0x07, 0x00, 0x12, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00,
  0x4c, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00, 0x59, 0x00, 0x00, 0x00,
  0x57, 0x00, 0x00, 0x00, 0xb3, 0x00, 0x05, 0x00, 0x15, 0x00, 0x00, 0x00,
  0x5a, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00,
  0xf6, 0x00, 0x04, 0x00, 0x5b, 0x00, 0x00, 0x00, 0x57, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00, 0x5a, 0x00, 0x00, 0x00,
  0x57, 0x00, 0x00, 0x00, 0x5b, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x57, 0x00, 0x00, 0x00, 0x50, 0x00, 0x05, 0x00, 0x2c, 0x00, 0x00, 0x00,
  0x5c, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x05, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x5d, 0x00, 0x00, 0x00,
  0x37, 0x00, 0x00, 0x00, 0x5c, 0x00, 0x00, 0x00, 0x6f, 0x00, 0x04, 0x00,
  0x1d, 0x00, 0x00, 0x00, 0x5e, 0x00, 0x00, 0x00, 0x5d, 0x00, 0x00, 0x00,
  0x81, 0x00, 0x05, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x5f, 0x00, 0x00, 0x00,
  0x5e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x6f, 0x00, 0x04, 0x00,
  0x1d, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00,
  0x88, 0x00, 0x05, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x61, 0x00, 0x00, 0x00,
  0x5f, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x56, 0x00, 0x05, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x64, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00,
  0x58, 0x00, 0x07, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x65, 0x00, 0x00, 0x00,
  0x64, 0x00, 0x00, 0x00, 0x61, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x19, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x06, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x66, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x58, 0x00, 0x00, 0x00, 0xc3, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x67, 0x00, 0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
  0xc7, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0x66, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00,
  0x25, 0x00, 0x00, 0x00, 0x69, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x07, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x6a, 0x00, 0x00, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00,
  0x69, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x6b, 0x00, 0x00, 0x00, 0x6a, 0x00, 0x00, 0x00, 0x50, 0x00, 0x07, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00,
  0x6b, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x08, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00, 0x65, 0x00, 0x00, 0x00,
  0x6c, 0x00, 0x00, 0x00, 0x56, 0x00, 0x00, 0x00, 0x80, 0x00, 0x05, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x59, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00, 0x55, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x5b, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x06, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0xc3, 0x00, 0x05, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x6e, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x6f, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x25, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00,
  0x6f, 0x00, 0x00, 0x00, 0x41, 0x00, 0x07, 0x00, 0x2f, 0x00, 0x00, 0x00,
  0x71, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x6e, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x72, 0x00, 0x00, 0x00, 0x71, 0x00, 0x00, 0x00,
  0x50, 0x00, 0x07, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x73, 0x00, 0x00, 0x00,
  0x72, 0x00, 0x00, 0x00, 0x72, 0x00, 0x00, 0x00, 0x72, 0x00, 0x00, 0x00,
  0x72, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0x11, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x56, 0x00, 0x00, 0x00, 0x73, 0x00, 0x00, 0x00, 0x4e, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x4f, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x4f, 0x00, 0x00, 0x00, 0x80, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x4d, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x53, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x74, 0x00, 0x00, 0x00, 0x4e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00, 0x75, 0x00, 0x00, 0x00,
  0x4e, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x76, 0x00, 0x00, 0x00, 0x4e, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x50, 0x00, 0x07, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0x77, 0x00, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00, 0x75, 0x00, 0x00, 0x00,
  0x76, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00,
  0x2d, 0x00, 0x00, 0x00, 0x78, 0x00, 0x00, 0x00, 0x37, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x63, 0x00, 0x05, 0x00, 0x79, 0x00, 0x00, 0x00,
  0x78, 0x00, 0x00, 0x00, 0x77, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x34, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x34, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00
};
