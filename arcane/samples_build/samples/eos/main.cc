#include <arcane/Directory.h>
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
  auto& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName("EOS");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  // Positionne le chemin absolu de la DLL C# en considérant qu'elle
  // se trouve dans le répertoire de l'exécutable
  auto& dotnet_info = ArcaneLauncher::dotNetRuntimeInitialisationInfo();
  String exe_dir = ArcaneLauncher::getExeDirectory();
  dotnet_info.setMainAssemblyName(Directory(exe_dir).file("PerfectGas.dll"));
  return ArcaneLauncher::run();
}
