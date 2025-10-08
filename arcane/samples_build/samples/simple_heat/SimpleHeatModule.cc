// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleHeatModule.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Module simplifié d'équation de la chaleur explicite.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleHeat_axl.h"

#include <arcane/core/ITimeLoopMng.h>
#include <arcane/core/IItemFamily.h>
#include <arcane/core/IndexedItemConnectivityView.h>
#include <arcane/core/IMesh.h>
#include <arcane/core/UnstructuredMeshConnectivity.h>
#include <arcane/core/ItemPrinter.h>
#include <arcane/core/SimpleSVGMeshExporter.h>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module SimpleHeatModule.
 */
class SimpleHeatModule
: public ArcaneSimpleHeatObject
{
 public:

  explicit SimpleHeatModule(const ModuleBuildInfo& mbi);

 public:

  /*!
   * \brief Méthode appelée à chaque itération.
   */
  void compute() override;
  /*!
   * \brief Méthode appelée lors de l'initialisation.
   */
  void startInit() override;

  /** Retourne le numéro de version du module */
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 private:

  void _applyBoundaryCondition();

  Int32 _getNeighbourFaceIndex(CellLocalId cell, CellLocalId neighbour_cell);
  Int32 _getCurrentFaceIndex(CellLocalId cell, FaceLocalId face);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleHeatModule::
SimpleHeatModule(const ModuleBuildInfo& mbi)
: ArcaneSimpleHeatObject(mbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHeatModule::
compute()
{
  info() << "Module SimpleHeatModule COMPUTE";

  // Stop code after 10 iterations
  if (m_global_iteration() > options()->nbIteration()) {
    subDomain()->timeLoopMng()->stopComputeLoop(true);
    return;
  }

  // Mise a jour de la temperature aux noeuds en prenant la moyenne
  // valeurs aux mailles voisines
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;
    Real sumt = 0;
    for (Cell cell : node.cells())
      sumt += m_cell_temperature[cell];
    m_node_temperature[inode] = sumt / node.nbCell();
  }
  m_node_temperature.synchronize();

  // Mise a jour de la temperature aux mailles en prenant la moyenne
  // des valeurs aux noeuds voisins
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    Real sumt = 0;
    for (Node node : cell.nodes())
      sumt += m_node_temperature[node];
    m_cell_temperature[icell] = sumt / cell.nbNode();
  }

  _applyBoundaryCondition();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHeatModule::
startInit()
{
  info() << "Module SimpleHeatModule INIT";
  m_cell_temperature.fill(0.0);
  m_node_temperature.fill(0.0);

  // Initialise le pas de temps à une valeur fixe
  m_global_deltat = 1.0;

  const bool is_verbose = false;

  // Calcule le centre des mailles
  VariableNodeReal3& nodes_coords(mesh()->nodesCoordinates());
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Int32 nb_node = cell.nbNode();
    Real3 center;
    for (NodeLocalId node : cell.nodeIds()) {
      center += nodes_coords[node];
    }
    center /= nb_node;
    m_cell_center[cell] = center;
    if (is_verbose)
      info() << "Cell " << ItemPrinter(cell) << " center=" << center;
  }

  _applyBoundaryCondition();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHeatModule::
_applyBoundaryCondition()
{
  // Positionne la température sur une partie du maillage
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Real3 center = m_cell_center[cell];
    Real norm = center.squareNormL2();
    if (norm < 0.1) {
      m_cell_temperature[cell] = 25000.0;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_SIMPLEHEAT(SimpleHeatModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
