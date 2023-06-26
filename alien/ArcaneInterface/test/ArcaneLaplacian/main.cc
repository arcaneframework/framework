#include <iostream>
#include "arcane_packages.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/launcher/ArcaneLauncher.h"
#include <arcane/utils/VersionInfo.h>
#include <arcane/impl/ArcaneMain.h>

using namespace Arcane;

int
main(int argc, char* argv[])
{
  int r = 0;
  ArcaneMain::arcaneInitialize();
  {
    ApplicationInfo app_info(&argc, &argv, "Laplacian", VersionInfo(1, 0, 0));
    r = ArcaneMain::arcaneMain(app_info);
  }
  ArcaneMain::arcaneFinalize();
  return r;
}
