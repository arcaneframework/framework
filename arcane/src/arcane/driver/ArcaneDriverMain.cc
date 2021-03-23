#include "arcane/impl/ArcaneMain.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/ApplicationInfo.h"

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ApplicationInfo& app_info = ArcaneMain::defaultApplicationInfo();
  app_info.setCommandLineArguments(CommandLineArguments(&argc,&argv));
  app_info.setCodeName("ArcaneDriver");
  app_info.setCodeVersion(VersionInfo(1,0,0));
  app_info.addDynamicLibrary("arcane_mpi");
  app_info.addDynamicLibrary("arcane_ios");
  app_info.addDynamicLibrary("arcane_std");
  app_info.addDynamicLibrary("arcane_mesh");
  app_info.addDynamicLibrary("arcane_cea");
  return ArcaneMain::run();
}
