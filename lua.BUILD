# Description:
#   Build rule for Lua 5.1.
#   Compiler and linker flags found with `pkg-config --cflags --libs lua5.1`.
#   The package name and the resulting flags may vary from platform to platform,
#   cf. 'How to build DeepMind Lab' in build.md.

cc_library(
    name = "lua",
    hdrs = glob(["include/*.h"]),
    includes = ["include/"],
    linkopts = ["-L/mnt/lustre/lana/zaytsev2/lustre/usr/lib -llua"],
    visibility = ["//visibility:public"],
)
