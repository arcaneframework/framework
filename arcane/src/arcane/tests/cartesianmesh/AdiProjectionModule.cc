// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AdiProjectionModule.cc                                      (C) 2000-2026 */
/*                                                                           */
/* Module for testing a projection on a Cartesian mesh.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeLoopService.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/MeshAreaAccessor.h"
#include "arcane/core/MeshArea.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/VariableViews.h"

#include "arcane/tests/cartesianmesh/AdiProjection_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AdiProjectionModule
: public ArcaneAdiProjectionObject
{
 public:

  explicit AdiProjectionModule(const ModuleBuildInfo& mb);
  ~AdiProjectionModule();

 public:

  virtual VersionInfo versionInfo() const { return VersionInfo(1, 1, 1); }

 public:

  void copyEulerianCoordinates();
  void cartesianHydroMain();
  virtual void cartesianHydroStartInit();

 public:

  static void staticInitialize(ISubDomain* sd);

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;

 private:

  void evolvePrimalUpwindedVariables(Integer direction);
  void evolveDualUpwindedVariables(Integer direction);
  void computePressure();
  void computePressureGradient(Integer direction);
  void checkNodalMassConservation();
  void copyCurrentVariablesToOldVariables();

  void computePrimalMassFluxInner(Integer direction);

  void computeDualMassFluxInner(Integer direction);
  void prepareLagrangianVariables();
  void checkLagrangianVariablesConsistency();

  void _evolveDualUpwindedVariables1();
  void _evolvePrimalUpwindedVariablesV2(Integer direction);

 public:

  // Public functions for CUDA
  void computePrimalMassFluxBoundary(Integer direction);
  void computeDualMassFluxBoundary(Integer direction);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ADIPROJECTION(AdiProjectionModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AdiProjectionModule::
AdiProjectionModule(const ModuleBuildInfo& mb)
: ArcaneAdiProjectionObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AdiProjectionModule::
~AdiProjectionModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
copyCurrentVariablesToOldVariables()
{
  ENUMERATE_CELL (current_cell, allCells()) {

    m_old_density[current_cell] = m_density[current_cell];
  }

  ENUMERATE_NODE (current_node, allNodes()) {

    m_old_velocity[current_node] = m_velocity[current_node];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
prepareLagrangianVariables()
{
  // Node positions after the Lagrange phase (= node positions
  // stored in the mesh).
  VariableNodeReal3& nodes_coord = defaultMesh()->nodesCoordinates();
  m_lagrangian_coordinates.copy(nodes_coord);

  ENUMERATE_ (Node, inode, allNodes()) {

    // Node displacement velocity (valid in predictor-corrector).
    m_lagrangian_velocity[inode] = 0.5 * (m_old_velocity[inode] + m_velocity[inode]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
checkLagrangianVariablesConsistency()
{
  // Verify that: u * dt = x_lagrange - x_euler.

  Real residu = 0.0;

  ENUMERATE_NODE (current_node, allNodes()) {
  }

  info() << "Test of Lagrangian velocity/Lagrangian position consistency: residue="
         << residu << "\n";

  // mass = rho * volume.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
computePrimalMassFluxInner(Integer direction)
{
  ARCANE_UNUSED(direction);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
computePrimalMassFluxBoundary(Integer direction)
{
  info() << A_FUNCINFO;

  auto queue = makeQueue(acceleratorMng()->defaultRunner());
  auto command = makeCommand(queue);

  auto inout_mass_flux_right = viewInOut(command, m_mass_flux_right);
  auto inout_mass_flux_left = viewInOut(command, m_mass_flux_left);

  CellDirectionMng cdm(m_cartesian_mesh->cellDirection(direction));

  // Calculation of mass fluxes for boundary cells in the direction of calculation.
  command << RUNCOMMAND_ENUMERATE (Cell, current_cell, cdm.outerCells())
  {
    // For left cell/right cell.
    DirCellLocalId cc(cdm.dirCellId(current_cell));

    CellLocalId right_cell = cc.next();
    CellLocalId left_cell = cc.previous();

    if (left_cell.isNull()) {
      // Left boundary.

      inout_mass_flux_right[current_cell] = inout_mass_flux_left[right_cell];
      inout_mass_flux_left[current_cell] = inout_mass_flux_right[current_cell];
    }
    else if (right_cell.isNull()) {
      // Right boundary.

      inout_mass_flux_left[current_cell] = inout_mass_flux_right[left_cell];
      inout_mass_flux_right[current_cell] = inout_mass_flux_left[current_cell];
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
computeDualMassFluxInner(Integer direction)
{
  ARCANE_UNUSED(direction);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
computeDualMassFluxBoundary(Integer direction)
{
  NodeDirectionMng ndm(m_cartesian_mesh->nodeDirection(direction));

  auto queue = makeQueue(acceleratorMng()->defaultRunner());
  auto command = makeCommand(queue);

  auto inout_nodal_mass_flux_right = viewInOut(command, m_nodal_mass_flux_right);
  auto inout_nodal_mass_flux_left = viewInOut(command, m_nodal_mass_flux_left);

  // Calculation of mass fluxes for boundary cells in the direction of calculation.
  command << RUNCOMMAND_ENUMERATE (Node, current_node, ndm.outerNodes())
  {
    // For left cell/right cell.
    DirNodeLocalId cc(ndm.dirNodeId(current_node));

    NodeLocalId right_node = cc.next();
    NodeLocalId left_node = cc.previous();

    if (left_node.isNull()) {
      // Left boundary.

      inout_nodal_mass_flux_left[current_node] = inout_nodal_mass_flux_left[right_node];
      inout_nodal_mass_flux_right[current_node] = inout_nodal_mass_flux_right[right_node];
    }
    else if (right_node.isNull()) {
      // Right boundary.

      inout_nodal_mass_flux_left[current_node] = inout_nodal_mass_flux_left[left_node];
      inout_nodal_mass_flux_right[current_node] = inout_nodal_mass_flux_right[left_node];
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Calculation of primal quantities: density, internal energy.
void AdiProjectionModule::
evolvePrimalUpwindedVariables(Integer direction)
{
  if (0) {
    _evolvePrimalUpwindedVariablesV2(direction);
    return;
  }
  // Hardcoded for now.
  const Real time_step = m_global_deltat();
  const Real dx = 0.005;

  CellDirectionMng cdm(m_cartesian_mesh->cellDirection(direction));

  // Loop over inner cells.

  ENUMERATE_CELL (current_cell, cdm.innerCells()) {

    // For left cell/right cell.
    DirCell cc(cdm.cell(*current_cell));

    CellLocalId right_cell = cc.next();
    CellLocalId left_cell = cc.previous();
    //Cell::Index right_cell = right_cell_c;
    //Cell::Index left_cell = left_cell_c;

    // For directional cell/node.
    DirCellNode cn(cdm.cellNode(*current_cell));

    // Temporary for 1D. Waiting for Cell/directional face connectivity.

    const Real3 left_face_velocity = m_lagrangian_velocity[cn.previousLeft()];
    const Real left_face_velocity_dir = left_face_velocity.x;

    const Real3 right_face_velocity = m_lagrangian_velocity[cn.nextLeft()];
    const Real right_face_velocity_dir = right_face_velocity.x;

    // Calculation of signs for decentralization.
    const Real sign_left = (left_face_velocity_dir > 0.0 ? 1.0 : -1.0);
    const Real sign_right = (right_face_velocity_dir > 0.0 ? 1.0 : -1.0);

    Real m_nrj_left_cell = m_nrj[left_cell];
    Real m_nrj_right_cell = m_nrj[right_cell];
    Real m_nrj_current_cell = m_nrj[current_cell];

    const Real dmass_left = 0.5 * left_face_velocity_dir *
    ((m_density[current_cell] + m_density[left_cell]) -
     sign_left * (m_density[current_cell] - m_density[left_cell]));

    const Real dmass_right = 0.5 * right_face_velocity_dir *
    ((m_density[right_cell] + m_density[current_cell]) -
     sign_right * (m_density[right_cell] - m_density[current_cell]));

    m_mass_flux_left[current_cell] = dmass_left;
    m_mass_flux_right[current_cell] = dmass_right;

    m_density[current_cell] =
    m_density[current_cell] - time_step * (dmass_right - dmass_left) / dx;

    // Decentralization of internal energy.
    const Real nrj_left = 0.5 *
    ((m_nrj_current_cell + m_nrj_left_cell) -
     sign_left * (m_nrj_current_cell - m_nrj_left_cell));

    const Real nrj_right = 0.5 *
    ((m_nrj_right_cell + m_nrj_current_cell) -
     sign_right * (m_nrj_right_cell - m_nrj_current_cell));

    Real nrj_current_cell = m_old_density[current_cell] * m_nrj_current_cell -
    time_step * (nrj_right * dmass_right - nrj_left * dmass_left) / dx;

    // PdV source term.
    nrj_current_cell = nrj_current_cell -
    time_step * m_pressure[current_cell] * (right_face_velocity_dir - left_face_velocity_dir) / dx;

    if (m_density[current_cell] != 0.0) {

      nrj_current_cell /= m_density[current_cell];
    }
    else {

      info() << "Error, zero density.\n";
      std::abort();
    }
    m_nrj[current_cell] = nrj_current_cell;
  }

  computePrimalMassFluxBoundary(direction);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Calculation of primal quantities: density, internal energy.
void AdiProjectionModule::
_evolvePrimalUpwindedVariablesV2(Integer direction)
{
  // Hardcoded for now.
  const Real time_step = m_global_deltat();
  const Real dx = 0.005;

  CellDirectionMng cdm(m_cartesian_mesh->cellDirection(direction));

  // Loop over inner cells.

  VariableCellReal& nrj = m_nrj;
  VariableCellReal& old_density = m_old_density;
  VariableCellReal& density = m_density;
  VariableCellReal& pressure = m_pressure;

  ENUMERATE_CELL (i_current_cell, cdm.innerCells()) {

    // For left cell/right cell.
    DirCell cc(cdm.cell(*i_current_cell));

    Cell right_cell_c = cc.next();
    Cell left_cell_c = cc.previous();

    CellLocalId current_cell = *i_current_cell;
    Cell right_cell = right_cell_c;
    Cell left_cell = left_cell_c;

    // For cell/directional node.
    DirCellNode cn(cdm.cellNode(*i_current_cell));

    Real nrj_left_cell = nrj[left_cell];
    Real nrj_right_cell = nrj[right_cell];
    Real nrj_current_cell = nrj[current_cell];

    // Temporary for 1D. Waiting for Cell/directional face connectivity.

    const Real3 left_face_velocity = m_lagrangian_velocity[cn.previousLeft()];
    const Real left_face_velocity_dir = left_face_velocity.x;

    const Real3 right_face_velocity = m_lagrangian_velocity[cn.nextLeft()];
    const Real right_face_velocity_dir = right_face_velocity.x;

    // Calculate signs for decentralization.
    const Real sign_left = (left_face_velocity_dir > 0.0 ? 1.0 : -1.0);
    const Real sign_right = (right_face_velocity_dir > 0.0 ? 1.0 : -1.0);

    const Real dmass_left = 0.5 * left_face_velocity_dir *
    ((density[current_cell] + density[left_cell]) -
     sign_left * (density[current_cell] - density[left_cell]));

    const Real dmass_right = 0.5 * right_face_velocity_dir *
    ((density[right_cell] + density[current_cell]) -
     sign_right * (density[right_cell] - density[current_cell]));

    m_mass_flux_left[i_current_cell] = dmass_left;
    m_mass_flux_right[i_current_cell] = dmass_right;

    density[current_cell] = density[current_cell] - time_step * (dmass_right - dmass_left) / dx;

    // Decentralization of internal energy.
    const Real nrj_left = 0.5 * ((nrj_current_cell + nrj_left_cell) - sign_left * (nrj_current_cell - nrj_left_cell));

    const Real nrj_right = 0.5 * ((nrj_right_cell + nrj_current_cell) - sign_right * (nrj_right_cell - nrj_current_cell));

    nrj_current_cell = old_density[current_cell] * nrj_current_cell - time_step * (nrj_right * dmass_right - nrj_left * dmass_left) / dx;

    // PdV source term.
    nrj_current_cell = nrj_current_cell - time_step * pressure[current_cell] * (right_face_velocity_dir - left_face_velocity_dir) / dx;

    nrj[current_cell] = nrj_current_cell;

    if (density[current_cell] != 0.0) {

      nrj[current_cell] /= density[current_cell];
    }
    else {

      info() << "Error, zero density.\n";
      std::abort();
    }
  }

  computePrimalMassFluxBoundary(direction);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
computePressureGradient(Integer direction)
{
  ENUMERATE_NODE (current_node, allNodes()) {

    // The pressure gradient is a temporary variable that can potentially be
    // accumulated. It is better to set it to 0.
    m_pressure_gradient[current_node] = 0.0;
  }

  CellDirectionMng cdm(m_cartesian_mesh->cellDirection(direction));

  ENUMERATE_CELL (current_cell, cdm.innerCells()) {

    // For left cell/right cell.
    DirCell cc(cdm.cell(*current_cell));

    //Cell right_cell = cc.next();
    Cell left_cell = cc.previous();

    // For cell/directional node.
    DirCellNode cn(cdm.cellNode(*current_cell));

    //GG: is it current_cell or right_cell?
    const Real current_pressure_gradient = m_pressure[current_cell] - m_pressure[left_cell];

    // Each point in the mesh (except at the boundaries) will have its pressure
    // gradient calculated 2 times, but that's okay...
    m_pressure_gradient[cn.previousLeft()] = current_pressure_gradient;
    m_pressure_gradient[cn.previousRight()] = current_pressure_gradient;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Calculation of dual quantities: velocity (momentum).
void AdiProjectionModule::
evolveDualUpwindedVariables(Integer direction)
{
  NodeDirectionMng ndm(m_cartesian_mesh->nodeDirection(direction));

  // Hardcoded for now.
  const Real time_step = m_global_deltat();
  const Real dx = 0.005;

  _evolveDualUpwindedVariables1();

  computeDualMassFluxBoundary(direction);

  computePressureGradient(direction);

  ENUMERATE_NODE (current_node, ndm.innerNodes()) {

    DirNode dir_node(ndm[current_node]);
    Node left_node = dir_node.previous();
    Node right_node = dir_node.next();

    const Real sign_left = (m_nodal_mass_flux_left[current_node] > 0.0 ? 1.0 : -1.0);
    const Real sign_right = (m_nodal_mass_flux_right[current_node] > 0.0 ? 1.0 : -1.0);

    const Real3 nodal_velocity_right =
    0.5 * ((m_old_velocity[right_node] + m_old_velocity[current_node]) - sign_right * (m_old_velocity[right_node] - m_old_velocity[current_node]));

    const Real3 nodal_velocity_left =
    0.5 * ((m_old_velocity[current_node] + m_old_velocity[left_node]) - sign_left * (m_old_velocity[current_node] - m_old_velocity[left_node]));

    m_lagrangian_velocity[current_node] =
    m_old_nodal_density[current_node] * m_lagrangian_velocity[current_node] -
    time_step * (m_nodal_mass_flux_right[current_node] * nodal_velocity_right - m_nodal_mass_flux_left[current_node] * nodal_velocity_left) / dx;

    m_lagrangian_velocity[current_node].x -= time_step * m_pressure_gradient[current_node] / dx;

    if (m_nodal_density[current_node] != 0.0) {

      m_lagrangian_velocity[current_node] /= m_nodal_density[current_node];
    }
    else {

      info() << "Problem: zero nodal density.\n";

      std::abort();
    }

    m_velocity[current_node] = m_lagrangian_velocity[current_node];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Calculation of dual quantities: velocity (momentum).
void AdiProjectionModule::
_evolveDualUpwindedVariables1()
{
  ENUMERATE_NODE (current_node, ownNodes()) {

    const Node& node = *current_node;

    const Integer nb_cells = node.nbCell();
    if (nb_cells == 0)
      ARCANE_FATAL("No cell attached to the node");

    // Nodal densities.

    Real nodal_density_sum = 0.0;
    Real old_nodal_density_sum = 0.0;
    Real nodal_mass_flux_right_accumulation = 0.0;
    Real nodal_mass_flux_left_accumulation = 0.0;

    for (Cell node_cell : node.cells()) {

      nodal_density_sum += m_density[node_cell];
      old_nodal_density_sum += m_old_density[node_cell];
      nodal_mass_flux_right_accumulation += m_mass_flux_right[node_cell];
      nodal_mass_flux_left_accumulation += m_mass_flux_left[node_cell];
    }

    m_nodal_density[current_node] = nodal_density_sum / nb_cells;
    m_old_nodal_density[current_node] = old_nodal_density_sum / nb_cells;
    m_nodal_mass_flux_right[current_node] = nodal_mass_flux_right_accumulation / nb_cells;
    m_nodal_mass_flux_left[current_node] = nodal_mass_flux_left_accumulation / nb_cells;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Application of the equation of state. Hardcoded (perfect gas, gamma=1.4)
// for now.
void AdiProjectionModule::
computePressure()
{

  ENUMERATE_CELL (current_cell, allCells()) {

    const Real gamma = 1.4;

    m_pressure[current_cell] =
    (gamma - 1.0) * m_density[current_cell] * m_nrj[current_cell];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// We must have nodal mass conservation (calculated at the moment
// of momentum decentralization, based on cell mass). This is a useful diagnostic.
void AdiProjectionModule::
checkNodalMassConservation()
{

  // Hardcoded for now.
  const Real time_step = m_global_deltat();
  const Real dx = 0.005;

  ENUMERATE_NODE (current_node, ownNodes()) {

    m_delta_mass[current_node] =
    m_nodal_density[current_node] - m_old_nodal_density[current_node] +
    time_step * (m_nodal_mass_flux_right[current_node] - m_nodal_mass_flux_left[current_node]) / dx;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// ATTENTION: call BEFORE the Lagrange phase...
void AdiProjectionModule::
copyEulerianCoordinates()
{
  VariableNodeReal3& nodes_coord = defaultMesh()->nodesCoordinates();
  m_eulerian_coordinates.copy(nodes_coord);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
cartesianHydroStartInit()
{
  m_global_deltat = 1.0;
  m_lagrangian_velocity.fill(Real3::zero());
  m_old_velocity.fill(Real3::zero());
  m_velocity.fill(Real3::zero());

  // Creation of directional connectivity info (= Cartesian).
  IMesh* mesh = defaultMesh();

  m_cartesian_mesh = ICartesianMesh::getReference(mesh, true);
  m_cartesian_mesh->computeDirections();

  // Initialize internal energy assuming a perfect gas.
  ENUMERATE_CELL (icell, allCells()) {
    Real pressure = m_pressure[icell];
    Real adiabatic_cst = 1.4;
    Real density = m_density[icell];
    m_nrj[icell] = pressure / ((adiabatic_cst - 1.) * density);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
cartesianHydroMain()
{

  copyEulerianCoordinates();

  copyCurrentVariablesToOldVariables();

  //prepareLagrangianVariables();

  //checkLagrangianVariablesConsistency();

  const Integer direction_x = 0;

  evolvePrimalUpwindedVariables(direction_x);

  evolveDualUpwindedVariables(direction_x);

  computePressure();

  checkNodalMassConservation();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("AdiProjectionTestLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("AdiProjection.CartesianHydroStartInit"));
    time_loop->setEntryPoints(ITimeLoop::WInit, clist);
  }

  /*{
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("AdiProjection.init"));
    time_loop->setEntryPoints(ITimeLoop::WInit,clist);
    }*/

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("AdiProjection.CartesianHydroMain"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop, clist);
  }

  {
    StringList clist;
    clist.add("AdiProjection");
    time_loop->setRequiredModulesName(clist);
    clist.clear();
    clist.add("ArcanePostProcessing");
    clist.add("ArcaneCheckpoint");
    time_loop->setOptionalModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
