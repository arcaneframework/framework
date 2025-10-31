// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AdiProjectionModule.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Module de test d'une projection sur maillage cartésien.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

#include "arcane/tests/AdiProjection_axl.h"

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

  // Fonctions publiques pour CUDA
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
  // Position des noeuds après la phase Lagrange (= position des
  // noeuds stockée dans le maillage).
  VariableNodeReal3& nodes_coord = defaultMesh()->nodesCoordinates();
  m_lagrangian_coordinates.copy(nodes_coord);

  ENUMERATE_ (Node, inode, allNodes()) {

    // Vitesse de déplacement des noeuds (valable en prédicteur correcteur).
    m_lagrangian_velocity[inode] = 0.5 * (m_old_velocity[inode] + m_velocity[inode]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdiProjectionModule::
checkLagrangianVariablesConsistency()
{
  // Vérifier que : u * dt = x_lagrange - x_euler.

  Real residu = 0.0;

  ENUMERATE_NODE (current_node, allNodes()) {
  }

  info() << "Test de coherence vitesses lagrangiennes/positions lagrangiennes : residu="
         << residu << "\n";

  // masse = rho * volume.
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

  // Calcul des flux de masse pour les mailles de bord dans la direction de calcul.
  command << RUNCOMMAND_ENUMERATE(Cell, current_cell, cdm.outerCells())
  {
    // Pour maille gauche/maille droite.
    DirCellLocalId cc(cdm.dirCellId(current_cell));

    CellLocalId right_cell = cc.next();
    CellLocalId left_cell = cc.previous();

    if (left_cell.isNull()) {
      // Frontière gauche.

      inout_mass_flux_right[current_cell] = inout_mass_flux_left[right_cell];
      inout_mass_flux_left[current_cell] = inout_mass_flux_right[current_cell];
    }
    else if (right_cell.isNull()) {
      // Frontière droite.

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

  // Calcul des flux de masse pour les mailles de bord dans la direction de calcul.
  command << RUNCOMMAND_ENUMERATE(Node, current_node, ndm.outerNodes())
  {
    // Pour maille gauche/maille droite.
    DirNodeLocalId cc(ndm.dirNodeId(current_node));

    NodeLocalId right_node = cc.next();
    NodeLocalId left_node = cc.previous();

    if (left_node.isNull()) {
      // Frontière gauche.

      inout_nodal_mass_flux_left[current_node] = inout_nodal_mass_flux_left[right_node];
      inout_nodal_mass_flux_right[current_node] = inout_nodal_mass_flux_right[right_node];
    }
    else if (right_node.isNull()) {
      // Frontière droite.

      inout_nodal_mass_flux_left[current_node] = inout_nodal_mass_flux_left[left_node];
      inout_nodal_mass_flux_right[current_node] = inout_nodal_mass_flux_right[left_node];
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Calcul des quantités primales : densité, énergie interne.
void AdiProjectionModule::
evolvePrimalUpwindedVariables(Integer direction)
{
  if (0) {
    _evolvePrimalUpwindedVariablesV2(direction);
    return;
  }
  // En dur pour l'instant.
  const Real time_step = m_global_deltat();
  const Real dx = 0.005;

  CellDirectionMng cdm(m_cartesian_mesh->cellDirection(direction));

  // Boucle sur les mailles intérieures.

  ENUMERATE_CELL (current_cell, cdm.innerCells()) {

    // Pour maille gauche/maille droite.
    DirCell cc(cdm.cell(*current_cell));

    CellLocalId right_cell = cc.next();
    CellLocalId left_cell = cc.previous();
    //Cell::Index right_cell = right_cell_c;
    //Cell::Index left_cell = left_cell_c;

    // Pour maille/noeud directionnel.
    DirCellNode cn(cdm.cellNode(*current_cell));

    // Temporaire pour le 1d. En attendant la connectivite Maille/face directionnelle.

    const Real3 left_face_velocity = m_lagrangian_velocity[cn.previousLeft()];
    const Real left_face_velocity_dir = left_face_velocity.x;

    const Real3 right_face_velocity = m_lagrangian_velocity[cn.nextLeft()];
    const Real right_face_velocity_dir = right_face_velocity.x;

    // Calcul des signes pour le décentrement.
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

    // Décentrement énergie interne.
    const Real nrj_left = 0.5 *
    ((m_nrj_current_cell + m_nrj_left_cell) -
     sign_left * (m_nrj_current_cell - m_nrj_left_cell));

    const Real nrj_right = 0.5 *
    ((m_nrj_right_cell + m_nrj_current_cell) -
     sign_right * (m_nrj_right_cell - m_nrj_current_cell));

    Real nrj_current_cell = m_old_density[current_cell] * m_nrj_current_cell -
    time_step * (nrj_right * dmass_right - nrj_left * dmass_left) / dx;

    // Terme source PdV.
    nrj_current_cell = nrj_current_cell -
    time_step * m_pressure[current_cell] * (right_face_velocity_dir - left_face_velocity_dir) / dx;

    if (m_density[current_cell] != 0.0) {

      nrj_current_cell /= m_density[current_cell];
    }
    else {

      info() << "Erreur, densite nulle.\n";
      std::abort();
    }
    m_nrj[current_cell] = nrj_current_cell;
  }

  computePrimalMassFluxBoundary(direction);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Calcul des quantités primales : densité, énergie interne.
void AdiProjectionModule::
_evolvePrimalUpwindedVariablesV2(Integer direction)
{
  // En dur pour l'instant.
  const Real time_step = m_global_deltat();
  const Real dx = 0.005;

  CellDirectionMng cdm(m_cartesian_mesh->cellDirection(direction));

  // Boucle sur les mailles intérieures.

  VariableCellReal& nrj = m_nrj;
  VariableCellReal& old_density = m_old_density;
  VariableCellReal& density = m_density;
  VariableCellReal& pressure = m_pressure;

  ENUMERATE_CELL (i_current_cell, cdm.innerCells()) {

    // Pour maille gauche/maille droite.
    DirCell cc(cdm.cell(*i_current_cell));

    Cell right_cell_c = cc.next();
    Cell left_cell_c = cc.previous();

    CellLocalId current_cell = *i_current_cell;
    Cell right_cell = right_cell_c;
    Cell left_cell = left_cell_c;

    // Pour maille/noeud directionnel.
    DirCellNode cn(cdm.cellNode(*i_current_cell));

    Real nrj_left_cell = nrj[left_cell];
    Real nrj_right_cell = nrj[right_cell];
    Real nrj_current_cell = nrj[current_cell];

    // Temporaire pour le 1d. En attendant la connectivite Maille/face directionnelle.

    const Real3 left_face_velocity = m_lagrangian_velocity[cn.previousLeft()];
    const Real left_face_velocity_dir = left_face_velocity.x;

    const Real3 right_face_velocity = m_lagrangian_velocity[cn.nextLeft()];
    const Real right_face_velocity_dir = right_face_velocity.x;

    // Calcul des signes pour le décentrement.
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

    // Décentrement énergie interne.
    const Real nrj_left = 0.5 * ((nrj_current_cell + nrj_left_cell) - sign_left * (nrj_current_cell - nrj_left_cell));

    const Real nrj_right = 0.5 * ((nrj_right_cell + nrj_current_cell) - sign_right * (nrj_right_cell - nrj_current_cell));

    nrj_current_cell = old_density[current_cell] * nrj_current_cell - time_step * (nrj_right * dmass_right - nrj_left * dmass_left) / dx;

    // Terme source PdV.
    nrj_current_cell = nrj_current_cell - time_step * pressure[current_cell] * (right_face_velocity_dir - left_face_velocity_dir) / dx;

    nrj[current_cell] = nrj_current_cell;

    if (density[current_cell] != 0.0) {

      nrj[current_cell] /= density[current_cell];
    }
    else {

      info() << "Erreur, densite nulle.\n";
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

    // Le gradient de pression est une variable temporaire qui peut a
    // priori être cumulée. Il vaut mieux la mettre à 0.
    m_pressure_gradient[current_node] = 0.0;
  }

  CellDirectionMng cdm(m_cartesian_mesh->cellDirection(direction));

  ENUMERATE_CELL (current_cell, cdm.innerCells()) {

    // Pour maille gauche/maille droite.
    DirCell cc(cdm.cell(*current_cell));

    //Cell right_cell = cc.next();
    Cell left_cell = cc.previous();

    // Pour maille/noeud directionnel.
    DirCellNode cn(cdm.cellNode(*current_cell));

    //GG: est-ce current_cell ou right_cell ?
    const Real current_pressure_gradient = m_pressure[current_cell] - m_pressure[left_cell];

    // Chaque point du maillage (sauf aux bords) aura son gradient de
    // pression calculé 2 fois, mais ca n'est pas grave...
    m_pressure_gradient[cn.previousLeft()] = current_pressure_gradient;
    m_pressure_gradient[cn.previousRight()] = current_pressure_gradient;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Calcul des quantités duales : vitesse (quantité de mouvement).
void AdiProjectionModule::
evolveDualUpwindedVariables(Integer direction)
{
  NodeDirectionMng ndm(m_cartesian_mesh->nodeDirection(direction));

  // En dur pour l'instant.
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

      info() << "Probleme : densite nodale nulle.\n";

      std::abort();
    }

    m_velocity[current_node] = m_lagrangian_velocity[current_node];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Calcul des quantités duales : vitesse (quantité de mouvement).
void AdiProjectionModule::
_evolveDualUpwindedVariables1()
{
  ENUMERATE_NODE (current_node, ownNodes()) {

    const Node& node = *current_node;

    const Integer nb_cells = node.nbCell();
    if (nb_cells == 0)
      ARCANE_FATAL("No cell attached to the node");

    // Densités nodales.

    Real nodal_density_sum = 0.0;
    Real old_nodal_density_sum = 0.0;
    Real nodal_mass_flux_right_accumulation = 0.0;
    Real nodal_mass_flux_left_accumulation = 0.0;

    for (Cell node_cell : node.cells() ) {

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

// Application de l'équation d'état. En dur (gaz parfaits, gamma=1.4)
// pour l'instant.
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

// On doit avoir conservation de la masse nodale (calculée au moment
// du décentrement de la quantité de mouvement, à partir de la masse
// aux mailles). C'est un diagnostic utile.
void AdiProjectionModule::
checkNodalMassConservation()
{

  // En dur pour l'instant.
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

// ATTENTION : à appeler AVANT la phase Lagrange...
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

  // Création des infos de connectivités directionnelles (= cartésiennes).
  IMesh* mesh = defaultMesh();

  m_cartesian_mesh = ICartesianMesh::getReference(mesh, true);
  m_cartesian_mesh->computeDirections();

  // Initialise l'énergie interne en supposant qu'on a un gaz parfait.
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
