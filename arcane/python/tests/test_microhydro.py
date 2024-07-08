import ArcanePython
import Arcane
import sys

from Arcane import ArcaneLauncher

import os

args = Arcane.CommandLineArguments.Create(sys.argv)
ArcaneLauncher.Init(args)

app_info = ArcaneLauncher.ApplicationInfo;
app_info.SetCodeName("ArcaneTest")
app_build_info = ArcaneLauncher.ApplicationBuildInfo;
app_build_info.AddDynamicLibrary("arcane_tests_lib")
exit(ArcaneLauncher.Run())
