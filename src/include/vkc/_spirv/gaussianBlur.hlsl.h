#if 0
; SPIR-V
; Version: 1.0
; Generator: Google spiregg; 0
; Bound: 141
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
               OpName %type_sampler "type.sampler"
               OpName %srcSampler "srcSampler"
               OpName %type_2d_image "type.2d.image"
               OpName %srcTex "srcTex"
               OpName %type_2d_image_0 "type.2d.image"
               OpName %dstImage "dstImage"
               OpName %type_ConstantBuffer_UBO "type.ConstantBuffer.UBO"
               OpMemberName %type_ConstantBuffer_UBO 0 "weights"
               OpName %ubo "ubo"
               OpName %type_RWStructuredBuffer_float "type.RWStructuredBuffer.float"
               OpName %writeBackWeights "writeBackWeights"
               OpName %main "main"
               OpName %type_sampled_image "type.sampled.image"
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %srcSampler DescriptorSet 0
               OpDecorate %srcSampler Binding 0
               OpDecorate %srcTex DescriptorSet 0
               OpDecorate %srcTex Binding 1
               OpDecorate %dstImage DescriptorSet 0
               OpDecorate %dstImage Binding 2
               OpDecorate %ubo DescriptorSet 0
               OpDecorate %ubo Binding 3
               OpDecorate %writeBackWeights DescriptorSet 0
               OpDecorate %writeBackWeights Binding 4
               OpMemberDecorate %type_ConstantBuffer_PushConstants 0 Offset 0
               OpDecorate %type_ConstantBuffer_PushConstants Block
               OpDecorate %_arr_v4float_uint_4 ArrayStride 16
               OpMemberDecorate %type_ConstantBuffer_UBO 0 Offset 0
               OpDecorate %type_ConstantBuffer_UBO Block
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %type_RWStructuredBuffer_float 0 Offset 0
               OpDecorate %type_RWStructuredBuffer_float BufferBlock
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %int_2 = OpConstant %int 2
      %int_3 = OpConstant %int 3
      %int_1 = OpConstant %int 1
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %float = OpTypeFloat 32
    %float_1 = OpConstant %float 1
    %float_0 = OpConstant %float 0
    %v4float = OpTypeVector %float 4
         %30 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
  %float_0_5 = OpConstant %float 0.5
    %v2float = OpTypeVector %float 2
         %33 = OpConstantComposite %v2float %float_0_5 %float_0_5
%type_ConstantBuffer_PushConstants = OpTypeStruct %int
%_ptr_PushConstant_type_ConstantBuffer_PushConstants = OpTypePointer PushConstant %type_ConstantBuffer_PushConstants
%type_sampler = OpTypeSampler
%_ptr_UniformConstant_type_sampler = OpTypePointer UniformConstant %type_sampler
%type_2d_image = OpTypeImage %float 2D 2 0 0 1 Unknown
%_ptr_UniformConstant_type_2d_image = OpTypePointer UniformConstant %type_2d_image
%type_2d_image_0 = OpTypeImage %float 2D 2 0 0 2 Rgba8
%_ptr_UniformConstant_type_2d_image_0 = OpTypePointer UniformConstant %type_2d_image_0
       %uint = OpTypeInt 32 0
     %uint_4 = OpConstant %uint 4
%_arr_v4float_uint_4 = OpTypeArray %v4float %uint_4
%type_ConstantBuffer_UBO = OpTypeStruct %_arr_v4float_uint_4
%_ptr_Uniform_type_ConstantBuffer_UBO = OpTypePointer Uniform %type_ConstantBuffer_UBO
%_runtimearr_float = OpTypeRuntimeArray %float
%type_RWStructuredBuffer_float = OpTypeStruct %_runtimearr_float
%_ptr_Uniform_type_RWStructuredBuffer_float = OpTypePointer Uniform %type_RWStructuredBuffer_float
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %45 = OpTypeFunction %void
      %v2int = OpTypeVector %int 2
     %v2uint = OpTypeVector %uint 2
%_ptr_PushConstant_int = OpTypePointer PushConstant %int
%_ptr_Uniform_float = OpTypePointer Uniform %float
%type_sampled_image = OpTypeSampledImage %type_2d_image
         %pc = OpVariable %_ptr_PushConstant_type_ConstantBuffer_PushConstants PushConstant
 %srcSampler = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant
     %srcTex = OpVariable %_ptr_UniformConstant_type_2d_image UniformConstant
   %dstImage = OpVariable %_ptr_UniformConstant_type_2d_image_0 UniformConstant
        %ubo = OpVariable %_ptr_Uniform_type_ConstantBuffer_UBO Uniform
%writeBackWeights = OpVariable %_ptr_Uniform_type_RWStructuredBuffer_float Uniform
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
     %uint_0 = OpConstant %uint 0
     %int_n2 = OpConstant %int -2
       %main = OpFunction %void None %45
         %52 = OpLabel
         %53 = OpLoad %v3uint %gl_GlobalInvocationID
               OpSelectionMerge %54 None
               OpSwitch %uint_0 %55
         %55 = OpLabel
         %56 = OpVectorShuffle %v2uint %53 %53 0 1
         %57 = OpBitcast %v2int %56
         %58 = OpAccessChain %_ptr_PushConstant_int %pc %int_0
         %59 = OpLoad %int %58
         %60 = OpSDiv %int %59 %int_2
         %61 = OpCompositeExtract %int %57 0
         %62 = OpSLessThanEqual %bool %61 %60
               OpSelectionMerge %63 None
               OpBranchConditional %62 %64 %63
         %64 = OpLabel
         %65 = OpShiftRightArithmetic %int %61 %int_2
         %66 = OpBitwiseAnd %int %61 %int_3
         %67 = OpBitcast %uint %66
         %68 = OpAccessChain %_ptr_Uniform_float %ubo %int_0 %65 %67
         %69 = OpLoad %float %68
         %70 = OpBitcast %uint %61
         %71 = OpAccessChain %_ptr_Uniform_float %writeBackWeights %int_0 %70
               OpStore %71 %69
               OpBranch %63
         %63 = OpLabel
         %72 = OpLoad %type_2d_image_0 %dstImage
         %73 = OpImageQuerySize %v2uint %72
         %74 = OpCompositeExtract %uint %73 0
         %75 = OpBitcast %int %74
         %76 = OpCompositeExtract %uint %73 1
         %77 = OpBitcast %int %76
         %78 = OpSGreaterThanEqual %bool %61 %75
         %79 = OpLogicalNot %bool %78
               OpSelectionMerge %80 None
               OpBranchConditional %79 %81 %80
         %81 = OpLabel
         %82 = OpCompositeExtract %int %57 1
         %83 = OpSGreaterThanEqual %bool %82 %77
               OpBranch %80
         %80 = OpLabel
         %84 = OpPhi %bool %true %63 %83 %81
               OpSelectionMerge %85 None
               OpBranchConditional %84 %86 %85
         %86 = OpLabel
               OpBranch %54
         %85 = OpLabel
         %87 = OpLoad %type_2d_image %srcTex
         %88 = OpImageQuerySizeLod %v2uint %87 %int_0
         %89 = OpCompositeExtract %uint %88 0
         %90 = OpBitcast %int %89
         %91 = OpCompositeExtract %uint %88 1
         %92 = OpBitcast %int %91
         %93 = OpCompositeConstruct %v2int %90 %92
         %94 = OpSDiv %int %59 %int_n2
               OpBranch %95
         %95 = OpLabel
         %96 = OpPhi %v4float %30 %85 %97 %98
         %99 = OpPhi %int %94 %85 %100 %98
        %101 = OpSLessThanEqual %bool %99 %60
               OpLoopMerge %102 %98 None
               OpBranchConditional %101 %103 %102
        %103 = OpLabel
               OpBranch %104
        %104 = OpLabel
        %105 = OpPhi %v4float %30 %103 %106 %107
        %108 = OpPhi %int %94 %103 %109 %107
        %110 = OpSLessThanEqual %bool %108 %60
               OpLoopMerge %111 %107 None
               OpBranchConditional %110 %107 %111
        %107 = OpLabel
        %112 = OpCompositeConstruct %v2int %108 %99
        %113 = OpIAdd %v2int %57 %112
        %114 = OpConvertSToF %v2float %113
        %115 = OpFAdd %v2float %114 %33
        %116 = OpConvertSToF %v2float %93
        %117 = OpFDiv %v2float %115 %116
        %118 = OpLoad %type_sampler %srcSampler
        %119 = OpSampledImage %type_sampled_image %87 %118
        %120 = OpImageSampleExplicitLod %v4float %119 %117 Lod %float_0
        %121 = OpExtInst %int %1 SAbs %108
        %122 = OpShiftRightArithmetic %int %121 %int_2
        %123 = OpBitwiseAnd %int %121 %int_3
        %124 = OpBitcast %uint %123
        %125 = OpAccessChain %_ptr_Uniform_float %ubo %int_0 %122 %124
        %126 = OpLoad %float %125
        %127 = OpVectorTimesScalar %v4float %120 %126
        %106 = OpFAdd %v4float %105 %127
        %109 = OpIAdd %int %108 %int_1
               OpBranch %104
        %111 = OpLabel
        %128 = OpExtInst %int %1 SAbs %99
        %129 = OpShiftRightArithmetic %int %128 %int_2
        %130 = OpBitwiseAnd %int %128 %int_3
        %131 = OpBitcast %uint %130
        %132 = OpAccessChain %_ptr_Uniform_float %ubo %int_0 %129 %131
        %133 = OpLoad %float %132
        %134 = OpVectorTimesScalar %v4float %105 %133
         %97 = OpFAdd %v4float %96 %134
               OpBranch %98
         %98 = OpLabel
        %100 = OpIAdd %int %99 %int_1
               OpBranch %95
        %102 = OpLabel
        %135 = OpCompositeExtract %float %96 0
        %136 = OpCompositeExtract %float %96 1
        %137 = OpCompositeExtract %float %96 2
        %138 = OpCompositeConstruct %v4float %135 %136 %137 %float_1
        %139 = OpBitcast %v2uint %57
        %140 = OpLoad %type_2d_image_0 %dstImage
               OpImageWrite %140 %139 %138 None
               OpBranch %54
         %54 = OpLabel
               OpReturn
               OpFunctionEnd

#endif

const unsigned char g_main[] = {
  0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x8d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00,
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
  0x06, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65, 0x2e, 0x73, 0x61, 0x6d,
  0x70, 0x6c, 0x65, 0x72, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x05, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x73, 0x72, 0x63, 0x53, 0x61, 0x6d, 0x70, 0x6c,
  0x65, 0x72, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x74, 0x79, 0x70, 0x65, 0x2e, 0x32, 0x64, 0x2e, 0x69, 0x6d, 0x61, 0x67,
  0x65, 0x00, 0x00, 0x00, 0x05, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x73, 0x72, 0x63, 0x54, 0x65, 0x78, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65, 0x2e, 0x32, 0x64, 0x2e,
  0x69, 0x6d, 0x61, 0x67, 0x65, 0x00, 0x00, 0x00, 0x05, 0x00, 0x05, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x64, 0x73, 0x74, 0x49, 0x6d, 0x61, 0x67, 0x65,
  0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x74, 0x79, 0x70, 0x65, 0x2e, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e,
  0x74, 0x42, 0x75, 0x66, 0x66, 0x65, 0x72, 0x2e, 0x55, 0x42, 0x4f, 0x00,
  0x06, 0x00, 0x05, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73, 0x00, 0x05, 0x00, 0x03, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x75, 0x62, 0x6f, 0x00, 0x05, 0x00, 0x0a, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65, 0x2e, 0x52, 0x57, 0x53,
  0x74, 0x72, 0x75, 0x63, 0x74, 0x75, 0x72, 0x65, 0x64, 0x42, 0x75, 0x66,
  0x66, 0x65, 0x72, 0x2e, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x07, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x77, 0x72, 0x69, 0x74,
  0x65, 0x42, 0x61, 0x63, 0x6b, 0x57, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73,
  0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x07, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65, 0x2e, 0x73, 0x61, 0x6d,
  0x70, 0x6c, 0x65, 0x64, 0x2e, 0x69, 0x6d, 0x61, 0x67, 0x65, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x11, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x16, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x02, 0x00, 0x18, 0x00, 0x00, 0x00, 0x29, 0x00, 0x03, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
  0x2b, 0x00, 0x04, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x1d, 0x00, 0x00, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x07, 0x00,
  0x1d, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x3f, 0x17, 0x00, 0x04, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x05, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00,
  0x1f, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x03, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x22, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x02, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x19, 0x00, 0x09, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
  0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x19, 0x00, 0x09, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x25, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00, 0x26, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x26, 0x00, 0x00, 0x00, 0x27, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x04, 0x00, 0x11, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00,
  0x27, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x03, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x11, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x28, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x03, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x03, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
  0x29, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x04, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x2b, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0x21, 0x00, 0x03, 0x00, 0x2d, 0x00, 0x00, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x2e, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
  0x2f, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x30, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x31, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x03, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
  0x22, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x3b, 0x00, 0x04, 0x00, 0x23, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x24, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
  0x25, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x3b, 0x00, 0x04, 0x00, 0x28, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x29, 0x00, 0x00, 0x00,
  0x0f, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
  0x2b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x26, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x33, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0x36, 0x00, 0x05, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x2d, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00, 0x36, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xfb, 0x00, 0x03, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x37, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x37, 0x00, 0x00, 0x00,
  0x4f, 0x00, 0x07, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x35, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00, 0x2e, 0x00, 0x00, 0x00,
  0x39, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x41, 0x00, 0x05, 0x00,
  0x30, 0x00, 0x00, 0x00, 0x3a, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x3b, 0x00, 0x00, 0x00, 0x3a, 0x00, 0x00, 0x00, 0x87, 0x00, 0x05, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x00, 0x00, 0x39, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xb3, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00,
  0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00,
  0x3e, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x40, 0x00, 0x00, 0x00, 0xc3, 0x00, 0x05, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x42, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x26, 0x00, 0x00, 0x00, 0x43, 0x00, 0x00, 0x00,
  0x42, 0x00, 0x00, 0x00, 0x41, 0x00, 0x07, 0x00, 0x31, 0x00, 0x00, 0x00,
  0x44, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x00, 0x00, 0x43, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x45, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x26, 0x00, 0x00, 0x00, 0x46, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x00, 0x00, 0x41, 0x00, 0x06, 0x00, 0x31, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x46, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x03, 0x00, 0x47, 0x00, 0x00, 0x00,
  0x45, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00, 0x3f, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x68, 0x00, 0x04, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x26, 0x00, 0x00, 0x00,
  0x4a, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00,
  0x4a, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x26, 0x00, 0x00, 0x00,
  0x4c, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00, 0x4d, 0x00, 0x00, 0x00,
  0x4c, 0x00, 0x00, 0x00, 0xaf, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x4e, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00,
  0xa8, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x00, 0x00,
  0x4e, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00, 0x4f, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x51, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x52, 0x00, 0x00, 0x00, 0x39, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0xaf, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00, 0x53, 0x00, 0x00, 0x00,
  0x52, 0x00, 0x00, 0x00, 0x4d, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x50, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x50, 0x00, 0x00, 0x00,
  0xf5, 0x00, 0x07, 0x00, 0x18, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00,
  0x19, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x53, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00, 0x55, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00, 0x54, 0x00, 0x00, 0x00,
  0x56, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x56, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00, 0x36, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x55, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x57, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x67, 0x00, 0x05, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00,
  0x57, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x26, 0x00, 0x00, 0x00, 0x59, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x5a, 0x00, 0x00, 0x00, 0x59, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x26, 0x00, 0x00, 0x00, 0x5b, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x5c, 0x00, 0x00, 0x00, 0x5b, 0x00, 0x00, 0x00, 0x50, 0x00, 0x05, 0x00,
  0x2e, 0x00, 0x00, 0x00, 0x5d, 0x00, 0x00, 0x00, 0x5a, 0x00, 0x00, 0x00,
  0x5c, 0x00, 0x00, 0x00, 0x87, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x5e, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x00, 0x33, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x5f, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x5f, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00, 0x1d, 0x00, 0x00, 0x00,
  0x60, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00,
  0x61, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00, 0x5e, 0x00, 0x00, 0x00,
  0x55, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00,
  0xb3, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00, 0x65, 0x00, 0x00, 0x00,
  0x63, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0xf6, 0x00, 0x04, 0x00,
  0x66, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xfa, 0x00, 0x04, 0x00, 0x65, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00,
  0x66, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x67, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x68, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x68, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00, 0x1d, 0x00, 0x00, 0x00,
  0x69, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00,
  0x6a, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00, 0x5e, 0x00, 0x00, 0x00,
  0x67, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00,
  0xb3, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00, 0x6e, 0x00, 0x00, 0x00,
  0x6c, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0xf6, 0x00, 0x04, 0x00,
  0x6f, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xfa, 0x00, 0x04, 0x00, 0x6e, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00,
  0x6f, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x6b, 0x00, 0x00, 0x00,
  0x50, 0x00, 0x05, 0x00, 0x2e, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00,
  0x6c, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00, 0x80, 0x00, 0x05, 0x00,
  0x2e, 0x00, 0x00, 0x00, 0x71, 0x00, 0x00, 0x00, 0x39, 0x00, 0x00, 0x00,
  0x70, 0x00, 0x00, 0x00, 0x6f, 0x00, 0x04, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x72, 0x00, 0x00, 0x00, 0x71, 0x00, 0x00, 0x00, 0x81, 0x00, 0x05, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x73, 0x00, 0x00, 0x00, 0x72, 0x00, 0x00, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x6f, 0x00, 0x04, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x74, 0x00, 0x00, 0x00, 0x5d, 0x00, 0x00, 0x00, 0x88, 0x00, 0x05, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x75, 0x00, 0x00, 0x00, 0x73, 0x00, 0x00, 0x00,
  0x74, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x76, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x56, 0x00, 0x05, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x77, 0x00, 0x00, 0x00, 0x57, 0x00, 0x00, 0x00,
  0x76, 0x00, 0x00, 0x00, 0x58, 0x00, 0x07, 0x00, 0x1d, 0x00, 0x00, 0x00,
  0x78, 0x00, 0x00, 0x00, 0x77, 0x00, 0x00, 0x00, 0x75, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x06, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00, 0xc3, 0x00, 0x05, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x7a, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x7b, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x26, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00,
  0x7b, 0x00, 0x00, 0x00, 0x41, 0x00, 0x07, 0x00, 0x31, 0x00, 0x00, 0x00,
  0x7d, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x7a, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x7e, 0x00, 0x00, 0x00, 0x7d, 0x00, 0x00, 0x00,
  0x8e, 0x00, 0x05, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x7f, 0x00, 0x00, 0x00,
  0x78, 0x00, 0x00, 0x00, 0x7e, 0x00, 0x00, 0x00, 0x81, 0x00, 0x05, 0x00,
  0x1d, 0x00, 0x00, 0x00, 0x6a, 0x00, 0x00, 0x00, 0x69, 0x00, 0x00, 0x00,
  0x7f, 0x00, 0x00, 0x00, 0x80, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x6d, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x68, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x6f, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x06, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x63, 0x00, 0x00, 0x00, 0xc3, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x81, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00,
  0xc7, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00, 0x82, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00,
  0x26, 0x00, 0x00, 0x00, 0x83, 0x00, 0x00, 0x00, 0x82, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x07, 0x00, 0x31, 0x00, 0x00, 0x00, 0x84, 0x00, 0x00, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x81, 0x00, 0x00, 0x00,
  0x83, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0x85, 0x00, 0x00, 0x00, 0x84, 0x00, 0x00, 0x00, 0x8e, 0x00, 0x05, 0x00,
  0x1d, 0x00, 0x00, 0x00, 0x86, 0x00, 0x00, 0x00, 0x69, 0x00, 0x00, 0x00,
  0x85, 0x00, 0x00, 0x00, 0x81, 0x00, 0x05, 0x00, 0x1d, 0x00, 0x00, 0x00,
  0x61, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x86, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x62, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x62, 0x00, 0x00, 0x00, 0x80, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x64, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x5f, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x66, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0x87, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x05, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x88, 0x00, 0x00, 0x00,
  0x60, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x89, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x50, 0x00, 0x07, 0x00, 0x1d, 0x00, 0x00, 0x00,
  0x8a, 0x00, 0x00, 0x00, 0x87, 0x00, 0x00, 0x00, 0x88, 0x00, 0x00, 0x00,
  0x89, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00,
  0x2f, 0x00, 0x00, 0x00, 0x8b, 0x00, 0x00, 0x00, 0x39, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x8c, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x63, 0x00, 0x05, 0x00, 0x8c, 0x00, 0x00, 0x00,
  0x8b, 0x00, 0x00, 0x00, 0x8a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x36, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x36, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00
};
