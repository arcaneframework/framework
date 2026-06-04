// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleHeatTestModule.cc                                     (C) 2000-2026 */
/*                                                                           */
/* Simplified module for the explicit heat equation.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/SimpleHeatTest_axl.h"

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ItemPrinter.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief SimpleHeatTestModule module.
 */
class SimpleHeatTestModule
: public ArcaneSimpleHeatTestObject
{
 public:

  explicit SimpleHeatTestModule(const ModuleBuildInfo& mbi);

 public:

  /*!
   * \brief Method called at each iteration.
   */
  void compute() override;
  /*!
   * \brief Method called during initialization.
   */
  void startInit() override;

  /** Returns the module version number */
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 private:

  void _applyBoundaryCondition();

  Int32 _getNeighbourFaceIndex(CellLocalId cell, CellLocalId neighbour_cell);
  Int32 _getCurrentFaceIndex(CellLocalId cell, FaceLocalId face);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleHeatTestModule::
SimpleHeatTestModule(const ModuleBuildInfo& mbi)
: ArcaneSimpleHeatTestObject(mbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHeatTestModule::
compute()
{
  info() << "Module SimpleHeatTestModule COMPUTE";

  // Stop code after 10 iterations
  if (m_global_iteration() > options()->nbIteration()) {
    subDomain()->timeLoopMng()->stopComputeLoop(true);
    return;
  }

  // Update the temperature at the nodes by taking the average
  // of the values of neighboring cells
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;
    Real sumt = 0;
    for (Cell cell : node.cells())
      sumt += m_cell_temperature[cell];
    m_node_temperature[inode] = sumt / node.nbCell();
  }
  m_node_temperature.synchronize();

  // Update the temperature at the cells by taking the average
  // of the values at neighboring nodes
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

void SimpleHeatTestModule::
startInit()
{
  info() << "Module SimpleHeatTestModule INIT";
  m_cell_temperature.fill(0.0);
  m_node_temperature.fill(0.0);

  // Initialize the time step to a fixed value
  m_global_deltat = 1.0;

  const bool is_verbose = false;

  // Calculate the center of the cells
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

void SimpleHeatTestModule::
_applyBoundaryCondition()
{
  // Sets the temperature on a part of the mesh
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

ARCANE_REGISTER_MODULE_SIMPLEHEATTEST(SimpleHeatTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
