// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UserFunctionModule.cc                                       (C) 2000-2023 */
/*                                                                           */
/* Module d'exemple pour les fonctions utilisateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/utils/Real3.h>

#include <arcane/core/ITimeLoopMng.h>
#include <arcane/core/IMesh.h>
#include <arcane/core/ICaseFunction.h>
#include <arcane/core/IStandardFunction.h>

#include "UserFunction_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Module UserFunction.
class UserFunctionModule
: public ArcaneUserFunctionObject
{
 public:

  explicit UserFunctionModule(const ModuleBuildInfo& mbi)
  : ArcaneUserFunctionObject(mbi)
  {}

 public:

  //! Method called at each iteration
  void compute() override;

  //! Method called at the beginning of the execution
  void startInit() override;

  //! Version of the module
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 private:

  IBinaryMathFunctor<Real, Real3, Real3>* m_node_velocity_functor = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UserFunctionModule::
compute()
{
  info() << "Module UserFunction COMPUTE";

  // Stop code after 10 iterations
  if (m_global_iteration() > 10)
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  // Loop over the cells and compute the center of each cell
  VariableNodeReal3& nodes_coordinates = defaultMesh()->nodesCoordinates();
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Real3 cell_center = Real3::zero();
    // Iteration over nodes of the cell
    for (Node node : cell.nodes()) {
      cell_center += nodes_coordinates[node];
    }
    cell_center /= cell.nbNode();
    m_cell_center[icell] = cell_center;
  }

  {
    // Compute NodeVelocity using external function
    const Real current_time = m_global_time();
    ENUMERATE_ (Node, inode, allNodes()) {
      Real3 position = nodes_coordinates[inode];
      m_node_velocity[inode] = m_node_velocity_functor->apply(current_time, position);
    }
  }

  {
    // Move nodes
    const Real current_deltat = m_global_deltat();
    ENUMERATE_ (Node, inode, allNodes()) {
      nodes_coordinates[inode] += m_node_velocity[inode] * current_deltat;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UserFunctionModule::
startInit()
{
  info() << "Module UserFunction INIT";
  m_global_deltat.assign(0.1);

  // Check we have user function for node velocity
  {
    ICaseFunction* opt_function = options()->nodeVelocity.function();
    IStandardFunction* scf = options()->nodeVelocity.standardFunction();
    if (!scf)
      ARCANE_FATAL("No standard case function for option 'node-velocity'");
    auto* functor = scf->getFunctorRealReal3ToReal3();
    if (!functor)
      ARCANE_FATAL("Standard function '{0}' is not convertible to f(Real,Real3) -> Real3", opt_function->name());
    m_node_velocity_functor = functor;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_USERFUNCTION(UserFunctionModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
