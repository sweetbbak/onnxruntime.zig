const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;

/// API docs: https://onnxruntime.ai/docs/api/c/struct_ort_api.html
pub const c_api = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

pub const OrtLoggingLevel = enum(u32) {
    verbose = c_api.ORT_LOGGING_LEVEL_VERBOSE,
    info = c_api.ORT_LOGGING_LEVEL_INFO,
    warning = c_api.ORT_LOGGING_LEVEL_WARNING,
    @"error" = c_api.ORT_LOGGING_LEVEL_ERROR,
    fatal = c_api.ORT_LOGGING_LEVEL_FATAL,
};

pub const ONNXTensorElementDataType = enum(u32) {
    undefined = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    bool = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    string = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, // maps to c++ type std::string
    f16 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    f32 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // maps to c type float
    f64 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, // maps to c type double
    bf16 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, // Non-IEEE floating-point format based on IEEE754 single-precision
    u8 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, // maps to c type uint8_t
    u16 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, // maps to c type uint16_t
    u32 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, // maps to c type uint32_t
    u64 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, // maps to c type uint64_t
    i8 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, // maps to c type int8_t
    i16 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, // maps to c type int16_t
    i32 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, // maps to c type int32_t
    i64 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, // maps to c type int64_t
    c64 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64, // complex with float32 real and imaginary components
    c128 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128, // complex with float64 real and imaginary components
};

pub const OrtAllocatorType = enum(i32) {
    invalid = c_api.OrtInvalidAllocator,
    device = c_api.OrtDeviceAllocator,
    arena = c_api.OrtArenaAllocator,
};

pub const OrtMemType = enum(i32) {
    /// The default allocator for execution provider
    default = c_api.OrtMemTypeDefault,
    /// Any CPU memory used by non-CPU execution provider
    cpu_input = c_api.OrtMemTypeCPUInput,
    /// CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
    cpu_output = c_api.OrtMemTypeCPUOutput,
};

pub const OnnxInstanceOpts = struct {
    model_path: [:0]const u8,
    log_level: OrtLoggingLevel,
    log_id: [:0]const u8,
    input_names: []const [:0]const u8,
    output_names: []const [:0]const u8,
    num_threads: u16 = 1,
};

pub const OnnxInstance = struct {
    const Self = @This();

    allocator: Allocator,
    ort_api: *const c_api.OrtApi,
    ort_env: *c_api.OrtEnv,
    session_opts: *c_api.OrtSessionOptions,
    session: *c_api.OrtSession,
    run_opts: ?*c_api.OrtRunOptions,
    input_names: []const [*:0]const u8,
    output_names: []const [*:0]const u8,
    mem_info: ?*c_api.OrtMemoryInfo = null,
    ort_inputs: ?[]*c_api.OrtValue = null,
    ort_outputs: ?[]?*c_api.OrtValue = null,

    pub fn init(
        allocator: Allocator,
        options: OnnxInstanceOpts,
    ) !*Self {
        const ort_api: *const c_api.OrtApi = c_api.OrtGetApiBase().*.GetApi.?(c_api.ORT_API_VERSION);

        const ort_env = try createEnv(ort_api, options);
        const session_opts = try createSessionOptions(ort_api);

        const res1 = ort_api.SetInterOpNumThreads.?(session_opts, options.num_threads);
        std.debug.assert(res1 == null);
        const res2 = ort_api.SetIntraOpNumThreads.?(session_opts, options.num_threads);
        std.debug.assert(res2 == null);

        const session = try createSession(ort_api, ort_env, session_opts, options);
        const run_opts = try createRunOptions(ort_api);

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        var input_names = try allocator.alloc([*:0]const u8, options.input_names.len);
        errdefer allocator.free(input_names);
        for (0..input_names.len) |i| {
            input_names[i] = options.input_names[i].ptr;
        }

        var output_names = try allocator.alloc([*:0]const u8, options.output_names.len);
        errdefer allocator.free(output_names);
        for (0..output_names.len) |i| {
            output_names[i] = options.output_names[i].ptr;
        }

        self.* = Self{
            .allocator = allocator,
            .ort_api = ort_api,
            .ort_env = ort_env,
            .session_opts = session_opts,
            .session = session,
            .run_opts = run_opts,
            .input_names = input_names,
            .output_names = output_names,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        // if (self.ort_inputs) |inputs| {
        //     for (inputs) |input| {
        //         self.ort_api.ReleaseValue.?(input);
        //     }
        // }
        //
        // if (self.ort_outputs) |outputs| {
        //     for (outputs) |output| {
        //         self.ort_api.ReleaseValue.?(output);
        //     }
        // }

        for (self.ort_inputs.?) |input| {
            self.ort_api.ReleaseValue.?(input);
        }

        for (self.ort_outputs.?) |output| {
            self.ort_api.ReleaseValue.?(output);
        }

        self.ort_api.ReleaseRunOptions.?(self.run_opts);
        self.ort_api.ReleaseSession.?(self.session);
        self.ort_api.ReleaseSessionOptions.?(self.session_opts);
        self.ort_api.ReleaseMemoryInfo.?(self.mem_info);
        self.ort_api.ReleaseEnv.?(self.ort_env);

        if (self.ort_inputs) |inputs| {
            self.allocator.free(inputs);
        }

        if (self.ort_outputs) |outputs| {
            self.allocator.free(outputs);
        }

        self.allocator.free(self.input_names);
        self.allocator.free(self.output_names);
        self.allocator.destroy(self);
    }

    pub fn initMemoryInfo(
        self: *Self,
        name: [:0]const u8,
        allocator_type: OrtAllocatorType,
        id: i32,
        mem_type: OrtMemType,
    ) !void {
        if (self.mem_info != null) @panic("Memory info already created");

        var mem_info: ?*c_api.OrtMemoryInfo = null;
        const status = self.ort_api.CreateMemoryInfo.?(
            name.ptr,
            @intFromEnum(allocator_type),
            id,
            @intFromEnum(mem_type),
            &mem_info,
        );

        try checkError(self.ort_api, status);
        self.mem_info = mem_info;
    }

    pub fn setManagedInputsOutputs(
        self: *Self,
        inputs: []*c_api.OrtValue,
        outputs: []?*c_api.OrtValue,
    ) void {
        // if (self.ort_inputs != null) @panic("Inputs already set");
        // if (self.ort_outputs != null) @panic("Outputs already set");

        self.ort_inputs = inputs;
        self.ort_outputs = outputs;
    }

    /// Create a tensor backed by a user supplied buffer.
    pub fn createTensorWithDataAsOrtValue(
        self: *Self,
        comptime T: type,
        data: []T,
        shape: []const i64,
        tensor_type: ONNXTensorElementDataType,
    ) !*c_api.OrtValue {
        var value: ?*c_api.OrtValue = null;
        const status = self.ort_api.CreateTensorWithDataAsOrtValue.?(
            self.mem_info.?,
            data.ptr,
            data.len * @sizeOf(T),
            shape.ptr,
            shape.len,
            @intFromEnum(tensor_type),
            &value,
        );

        try checkError(self.ort_api, status);
        return value.?;
    }

    /// Create a tensor
    pub fn createTensor(
        self: *Self,
        comptime T: type,
        data: []T,
        shape: []const i64,
        tensor_type: ONNXTensorElementDataType,
    ) !*c_api.OrtValue {
        var value: ?*c_api.OrtValue = null;
        const status = self.ort_api.CreateTensorAsOrtValue.?(
            self.mem_info.?,
            data.ptr,
            data.len * @sizeOf(T),
            shape.ptr,
            shape.len,
            @intFromEnum(tensor_type),
            &value,
        );

        try checkError(self.ort_api, status);
        return value.?;
    }

    pub fn run(self: *Self) !void {
        const status = self.ort_api.Run.?(
            self.session,
            self.run_opts,
            self.input_names.ptr,
            self.ort_inputs.?.ptr,
            self.ort_inputs.?.len,
            self.output_names.ptr,
            self.output_names.len,
            self.ort_outputs.?.ptr,
        );

        try checkError(self.ort_api, status);
    }

    pub fn checkError(
        ort_api: *const c_api.OrtApi,
        onnx_status: ?*c_api.OrtStatus,
    ) !void {
        if (onnx_status == null) return;
        defer ort_api.ReleaseStatus.?(onnx_status);

        const msg = ort_api.GetErrorMessage.?(onnx_status);
        std.debug.print("ONNX error: {s}\n", .{msg});

        return error.OnnxError;
    }

    fn createEnv(
        ort_api: *const c_api.OrtApi,
        options: OnnxInstanceOpts,
    ) !*c_api.OrtEnv {
        var ort_env: ?*c_api.OrtEnv = null;
        const status = ort_api.CreateEnv.?(
            @intFromEnum(options.log_level),
            options.log_id.ptr,
            &ort_env,
        );

        try checkError(ort_api, status);
        return ort_env.?;
    }

    fn createSessionOptions(
        ort_api: *const c_api.OrtApi,
    ) !*c_api.OrtSessionOptions {
        var ort_sess_opts: ?*c_api.OrtSessionOptions = null;
        const status = ort_api.CreateSessionOptions.?(&ort_sess_opts);

        try checkError(ort_api, status);
        return ort_sess_opts.?;
    }

    fn createSession(
        ort_api: *const c_api.OrtApi,
        ort_env: *c_api.OrtEnv,
        ort_sess_opts: *c_api.OrtSessionOptions,
        options: OnnxInstanceOpts,
    ) !*c_api.OrtSession {
        var ort_sess: ?*c_api.OrtSession = null;
        const status = ort_api.CreateSession.?(
            ort_env,
            options.model_path.ptr,
            ort_sess_opts,
            &ort_sess,
        );

        try checkError(ort_api, status);
        return ort_sess.?;
    }

    fn createRunOptions(
        ort_api: *const c_api.OrtApi,
    ) !*c_api.OrtRunOptions {
        var run_opts: ?*c_api.OrtRunOptions = null;
        const status = ort_api.CreateRunOptions.?(&run_opts);

        try checkError(ort_api, status);
        return run_opts.?;
    }

    pub fn isTensor(
        self: *Self,
        tensor: ?*c_api.OrtValue,
    ) !bool {
        var is_tensor: c_int = -1;
        const status = self.ort_api.IsTensor.?(tensor, &is_tensor);
        try Error(self.ort_api, status);
        return if (is_tensor == 1) true else false;
    }

    pub fn getTensorData(
        self: *Self,
        tensor: ?*c_api.OrtValue,
        output_data: anytype,
    ) !bool {
        try Error(
            self.ort_api,
            self.ort_api.GetTensorMutableData.?(tensor, @ptrCast(&output_data)),
        );
    }
};

pub fn getTensorShapeCount(
    allocator: Allocator,
    ort_api: *const c_api.OrtApi,
    tensor: ?*c_api.OrtValue,
) !i64 {
    var shape_info: ?*c_api.OrtTensorTypeAndShapeInfo = mem.zeroes(?*c_api.OrtTensorTypeAndShapeInfo);
    try Error(
        ort_api,
        ort_api.GetTensorTypeAndShape.?(tensor, &shape_info),
    );

    var num_dims: usize = 0;
    try Error(
        ort_api,
        ort_api.GetDimensionsCount.?(shape_info, &num_dims),
    );

    std.debug.print("num dims: {any}\n", .{num_dims});

    var output_dims: []i64 = try allocator.alloc(i64, num_dims);
    _ = &output_dims;
    // var output_dims_c: *i64 = @ptrCast(&output_dims);
    defer allocator.free(output_dims);

    try Error(
        ort_api,
        ort_api.GetDimensions.?(shape_info, @ptrCast(output_dims.ptr), num_dims),
    );
    const thing = output_dims[output_dims.len - 1];
    std.debug.print("output dims: {d}\n", .{thing});

    return output_dims[output_dims.len - 1];
}

pub fn getTensorNumDims(
    ort_api: *const c_api.OrtApi,
    tensor: ?*c_api.OrtValue,
) !usize {
    var shape_info: ?*c_api.OrtTensorTypeAndShapeInfo = mem.zeroes(?*c_api.OrtTensorTypeAndShapeInfo);
    try Error(
        ort_api,
        ort_api.GetTensorTypeAndShape.?(tensor, &shape_info),
    );

    var num_dims: usize = 0;
    try Error(
        ort_api,
        ort_api.GetDimensionsCount.?(shape_info, &num_dims),
    );

    return num_dims;
}

pub fn getTensorElementCount(
    allocator: Allocator,
    ort_api: *const c_api.OrtApi,
    tensor: ?*c_api.OrtValue,
) !usize {
    var shape_info: ?*c_api.OrtTensorTypeAndShapeInfo = mem.zeroes(?*c_api.OrtTensorTypeAndShapeInfo);
    try Error(
        ort_api,
        ort_api.GetTensorTypeAndShape.?(tensor, &shape_info),
    );

    var num_dims: usize = 0;
    try Error(
        ort_api,
        ort_api.GetDimensionsCount.?(shape_info, &num_dims),
    );
    std.debug.print("dims: {d}\n", .{num_dims});

    var output_dims: []i64 = try allocator.alloc(i64, num_dims);
    var output_dims_c: *i64 = @ptrCast(&output_dims);
    defer allocator.free(output_dims);

    try Error(
        ort_api,
        ort_api.GetDimensions.?(shape_info, @ptrCast(&output_dims_c), num_dims),
    );
    std.debug.print("output dims: {any}\n", .{output_dims});

    var elem_count: usize = 0;
    try Error(
        ort_api,
        ort_api.GetTensorShapeElementCount.?(shape_info, &elem_count),
    );

    var output_size: usize = 1;
    for (num_dims, 0..) |_, i| {
        std.debug.print("{d}\n", .{output_dims[i]});
        output_size *= @intCast(output_dims[i]);
    }

    std.debug.print("shape element count: {d}\n", .{output_size});
    ort_api.ReleaseTensorTypeAndShapeInfo.?(shape_info);
    return output_size;
}

/// check an OrtStatus error
pub fn Error(
    ort_api: *const c_api.OrtApi,
    onnx_status: ?*c_api.OrtStatus,
) !void {
    if (onnx_status == null) return;
    defer ort_api.ReleaseStatus.?(onnx_status);

    const msg = ort_api.GetErrorMessage.?(onnx_status);
    std.debug.print("ONNX error: {s}\n", .{msg});

    return error.OnnxError;
}

comptime {
    std.testing.refAllDecls(@This());
}
