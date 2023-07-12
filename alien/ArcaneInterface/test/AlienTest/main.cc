#include <iostream>
#include "arcane_packages.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/launcher/ArcaneLauncher.h"
#include <arcane/utils/VersionInfo.h>
#include <arcane/impl/ArcaneMain.h>


#include <arcane/impl/ArcaneMain.h>

#define USEALIEN_USE_MPI

#ifdef USEALIEN_USE_MPI
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#endif // USEALIEN_USE_MPI

using namespace Arcane;

int
main(int argc, char* argv[])
{
  int r = 0;
#ifdef USEALIEN_USE_MPI
  MPI_Init(&argc, &argv);
#endif // USEALIEN_USE_MPI
  ArcaneMain::arcaneInitialize();
  {
    ApplicationInfo app_info(&argc, &argv, "AlienTest", VersionInfo(1, 0, 0));
    r = ArcaneMain::arcaneMain(app_info);
  }
  ArcaneMain::arcaneFinalize();
#ifdef USEALIEN_USE_MPI
  MPI_Finalize();
#endif // USEALIEN_USE_MPI

  return r;
}
