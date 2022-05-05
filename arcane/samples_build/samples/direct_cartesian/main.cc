// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* main.cc                                                     (C) 2000-2022 */
/*                                                                           */
/* Main direct_cartesian sample.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/launcher/ArcaneLauncher.h>

#include <iostream>
#include <arcane/utils/ITraceMng.h>
#include <arcane/IMesh.h>
#include <arcane/ISubDomain.h>
#include <arcane/VariableTypes.h>

using namespace Arcane;

void
executeCode(ISubDomain* sd)
{
  // Récupère le maillage initial
  IMesh* mesh = sd->defaultMesh();
  // Récupère le gestionnaire de traces pour les affichages
  ITraceMng* tr = mesh->traceMng();
  // Affiche le nombre de mailles
  tr->info() << "NB_CELL=" << mesh->nbCell();

  // Coordonnées des noeuds du maillage
  const VariableNodeReal3& nodes_coordinates = mesh->nodesCoordinates();

  VariableCellReal3 cells_coordinates(VariableBuildInfo(mesh,"CellsCenter"));

  // Parcours l'ensemble des mailles et calcule leur centre à partir
  // des coordonnées des noeuds
  ENUMERATE_CELL(icell,mesh->allCells()){
    Cell cell = *icell;
    Integer nb_node = cell.nbNode();
    Real3 center;
    for( NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode ){
      center += nodes_coordinates[inode];
    }
    center /= nb_node;
    cells_coordinates[icell] = center;
  }
  tr->info() << "End of executeCode";  
}

int
main(int argc,char* argv[])
{
  // Le nom du fichier du jeu de données est le dernier argument de la ligne de commande.
  if (argc<2){
    std::cout << "Usage: DirectCartesian casefile.arc\n";
    return 1;
  }
  ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
  String case_file_name = argv[argc-1];
  // Déclare la fonction qui sera exécutée par l'appel à run()
  auto f = [=](DirectSubDomainExecutionContext& ctx) -> int
  {
    executeCode(ctx.subDomain());
    return 0;
  };
  // Exécute le fonctor 'f'.
  return ArcaneLauncher::run(f);
}
