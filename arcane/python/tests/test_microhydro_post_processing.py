import ArcanePython
import Arcane
import sys

from Arcane import ArcaneLauncher

import os
gvar0 = 1
def my_test_func(sd_context):
    print("HELLO context=",sd_context,flush=True)
    global gvar0
    gvar0 = gvar0 + 1

args = Arcane.CommandLineArguments.Create(sys.argv)
ArcaneLauncher.Init(args)

print("MY_TEST_BEGIN",flush=True)
app_info = ArcaneLauncher.ApplicationInfo;
app_info.SetCodeName("ArcaneTest")
app_build_info = ArcaneLauncher.ApplicationBuildInfo;
app_build_info.AddDynamicLibrary("arcane_tests_lib")
r = ArcaneLauncher.Run();
print("MY_TEST1_FINAL gvar0",gvar0)
exit(r)
