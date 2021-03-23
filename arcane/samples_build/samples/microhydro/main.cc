#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  auto& app_info = ArcaneLauncher::applicationInfo();
  app_info.setCommandLineArguments(CommandLineArguments(&argc,&argv));
  app_info.setCodeName("MicroHydro");
  return ArcaneLauncher::run();
}
