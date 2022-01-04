// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "PoissonModule.h"

#include <arcane/MathUtils.h>
#include <arcane/IParallelMng.h>
#include <arcane/ITimeLoopMng.h>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PoissonModule::
initTemperatures()
{
  // pour que les dépouillement ne se superpose pas
  m_global_deltat = 1;

  // initialisation de la temperature sur toutes les mailles
  ENUMERATE_CELL(icell, allCells()) {
    m_cell_temperature[icell] = options()->initTemperature();
  }

  // application de la temperature aux limites
  applyBoundaryConditions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PoissonModule::
propagateTemperatures()
{
  Real max_delta_cell_t = 0;

  // mise a jour de la temperature aux mailles
  ENUMERATE_CELL(icell, allCells()){
    // calcul de la nouvelle temperature
    Cell cell = *icell;
    Real sumt = 0;
    for (NodeEnumerator inode(cell.nodes()); inode(); ++inode)
      sumt += m_node_temperature[inode];
    Real new_cell_t = sumt / cell.nbNode();

    // on observe l'ecart
    Real delta_cell_t = math::abs(new_cell_t - m_cell_temperature[icell]);
    max_delta_cell_t = math::max(max_delta_cell_t, delta_cell_t);

    // mise a jour de la temperature
    m_cell_temperature[icell] = new_cell_t;
  }

  // mise a jour de la temperature aux noeuds
  ENUMERATE_NODE(inode, allNodes()){
    Node node = *inode;
    Real sumt = 0;
    for (CellEnumerator icell(node.cells()); icell(); ++icell)
      sumt += m_cell_temperature[icell];
    m_node_temperature[inode] = sumt / node.nbCell();
  }

  // Vu le calcul, la syncronisation de la temperature aux mailles est inutile
  // syncronisation de la temperature aux noeuds et réduction de l'écart
  m_node_temperature.synchronize();
  max_delta_cell_t = parallelMng()->reduce(Parallel::ReduceMax, max_delta_cell_t);

  // application des conditions aux limites
  applyBoundaryConditions();

  // test d'arret de la boucle en temps
  if (max_delta_cell_t<0.2)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PoissonModule::
applyBoundaryConditions()
{
  // boucle sur les conditions aux limites
  int nb_boundary_condition = options()->boundaryCondition.size();
  for (int i = 0; i < nb_boundary_condition; ++i) {
    FaceGroup face_group = options()->boundaryCondition[i]->surface();
    Real temperature = options()->boundaryCondition[i]->value();
    TypesPoisson::eBoundaryCondition type = options()->boundaryCondition[i]->type();

    // boucle sur les faces de la surface
    ENUMERATE_FACE(iface, face_group) {
      Face face = *iface;
      // boucle sur les noeuds de la face
      for (NodeEnumerator inode(face.nodes()); inode(); ++inode){
        switch (type){
        case TypesPoisson::Temperature:
          m_node_temperature[inode] = temperature;
          break;
        case TypesPoisson::Unknown:
          break;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_POISSON(PoissonModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
