#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/Exception.h"

#include "arcane/launcher/ArcaneLauncher.h"

using namespace Arcane;

namespace
{
int _runTest(int argc, char* argv[])
{
  ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
  auto& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName("AlienTest");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  return ArcaneLauncher::run();
}
}

int
main(int argc, char* argv[])
{
  int r = 0;
  int r2 = arcaneCallFunctionAndCatchException([&](){ r = _runTest(argc,argv); });
  if (r2!=0)
    return r2;
  return r;
}
