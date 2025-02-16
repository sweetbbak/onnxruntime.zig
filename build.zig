const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const onnx_dep = b.dependency("onnxruntime_linux_x64", .{});

    // installs onnxruntime.so libraries (without symlinks: https://github.com/ziglang/zig/pull/18619)
    const install_onnx_libs = b.addInstallDirectory(.{
        .source_dir = onnx_dep.path("lib"),
        .install_dir = .lib,
        .install_subdir = ".",
    });

    b.getInstallStep().dependOn(&install_onnx_libs.step);

    // copy and rename file
    const install_onnx = b.addInstallLibFile(
        onnx_dep.path("lib/libonnxruntime.so.1.17.1"),
        "libonnxruntime.so",
    );

    b.getInstallStep().dependOn(&install_onnx.step);

    const lib_mod = b.addModule(
        "zig-onnxruntime",
        .{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
        },
    );

    lib_mod.addIncludePath(onnx_dep.path("include"));
    lib_mod.addLibraryPath(onnx_dep.path("lib"));

    const onnx_lib_mod = b.addModule(
        "onnxruntime_lib",
        .{
            .root_source_file = onnx_dep.path("lib"),
        },
    );
    _ = onnx_lib_mod; // autofix

    // add the shared object directly
    lib_mod.addObjectFile(install_onnx.source);
}
