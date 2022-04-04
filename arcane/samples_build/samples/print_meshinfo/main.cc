#include <arcane/launcher/ArcaneLauncher.h>

#include <iostream>
#include <arcane/IMesh.h>
#include <arcane/MeshReaderMng.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  // Affiche le nombre de mailles du maillage
  // Le nom du maillage est le dernier argument de la ligne de commande.
  if (argc<2){
    std::cout << "Usage: print_meshinfo meshfile\n";
    return 1;
  }
  ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
  String mesh_name = argv[argc-1];
  auto f = [=](DirectExecutionContext& ctx) -> int
  {
    ISubDomain* sd = ctx.createSequentialSubDomain();
    MeshReaderMng mrm(sd);
    IMesh* mesh = mrm.readMesh("Mesh1",mesh_name);
    std::cout << "NB_CELL=" << mesh->nbCell() << "\n";
    return 0;
  };
  return ArcaneLauncher::run(f);
}
