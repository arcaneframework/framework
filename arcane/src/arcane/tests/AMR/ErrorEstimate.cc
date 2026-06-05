// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ErrorEstimate.cc                                            (C) 2000-2022 */
/*                                                                           */
/* Service of analytical solutions used to estimate AMR error.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/AMR/ErrorEstimate.h"

#include "arcane/core/VariableTypes.h"
#include "arcane/core/SharedVariable.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::
attachExactValue(Real fptr(const Real3& p))
{
  ARCANE_ASSERT((fptr != NULL), (""));
  m_exact_value = fptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::
attachExactGradient(Real3 fptr(const Real3& p))
{
  ARCANE_ASSERT((fptr != NULL), (""));
  m_exact_gradient = fptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::
attachExactHessian(Real3x3 fptr(const Real3& p))
{
  ARCANE_ASSERT((fptr != NULL), (""));
  m_exact_hessian = fptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::
computeSol(RealArray& sol, IMesh* mesh)
{
  //
  SharedVariableNodeReal3 orig_nodes_coords(mesh->sharedNodesCoordinates());
  // sol.reserve(mesh->allActiveCells().size());

  ENUMERATE_CELL (icell, mesh->allCells()) { // active_local_elements
    const Cell& cell = *icell;
    const Int32 nb_nodes = cell.nbNode();

    Real3 cellCenter(Real3::null());
    for (Integer i = 0; i < nb_nodes; ++i)
      cellCenter += orig_nodes_coords[cell.node(i)];
    cellCenter /= nb_nodes;

    // calculation of the error at the cell center
    Real exact_val = 0.0;
    if (m_exact_value) {
      //exact_val = m_exact_value(cellCenter);
      exact_val = m_exact_value(cellCenter);
      sol.add(exact_val);
    }
  } // end ownActiveCell enumerate
  std::cout << "SOL_SIZE=" << sol.size() << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::
computeError(RealArray& error, IMesh* mesh)
{
  this->_computeError(error, mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ErrorEstimate::
errorNorm(const NormType& norm)
{
  switch (norm) {
  case L2:
    return m_error_vals[0];
  case L_INF:
    return m_error_vals[1];
    // \todo to extend for other norms/semi-norms
  default:
    ARCANE_THROW(NotImplementedException, "Norm Type is not implemented!");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ErrorEstimate::
l2Error()
{
  // Return the L2 norm of the error.
  return m_error_vals[0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ErrorEstimate::
lInfError()
{

  // Return the infinity norm of the error.
  return m_error_vals[1];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::
_computeError(RealArray& error, IMesh* mesh)
{
  // Verification of the solution and its derivatives
  //ARCANE_ASSERT ((!(m_exact_value && m_exact_gradient)), (""));

  // initialization of the error to Zero before summation
  m_error_vals = Real3::null();

  // The tests are stationary
  // const Real time = 0.;//

  //
  SharedVariableNodeReal3 orig_nodes_coords(mesh->sharedNodesCoordinates());
  error.reserve(mesh->allActiveCells().size());
  error.fill(0.);
  ENUMERATE_CELL (icell, mesh->allCells()) { // active_local_elements
    Cell cell = *icell;
    const Int32 nb_nodes = cell.nbNode();

    Real3 cellCenter(Real3::null());
    for (Integer i = 0; i < nb_nodes; ++i)
      cellCenter += orig_nodes_coords[cell.node(i)];
    cellCenter /= nb_nodes;

    // calculation of the error at the cell center
    Real3 grad_exact_value(Real3::null());
    if (m_exact_gradient) {
      //exact_val = m_exact_value(cellCenter);
      grad_exact_value = m_exact_gradient(cellCenter);
    }

    const Real& val_error = pow(grad_exact_value[0], 2) + pow(grad_exact_value[1], 2) +
    pow(grad_exact_value[2], 2); // x**2 +y**2+z**2

    error.add(math::sqrt(val_error));
    // Assembly of the error
    m_error_vals[0] += val_error;
    Real norm = math::sqrt(val_error);

    if (m_error_vals[1] < norm) {
      m_error_vals[1] = norm;
    }

  } // end ownActiveCell enumerate

  // Add up the error values on all processors, except for the L-infty
  // norm, for which the maximum is computed.
  IParallelMng* pm = mesh->parallelMng();

  if (pm->commSize() > 1) {
    const Real& l2 = m_error_vals[0];
    const Real& l_inf = m_error_vals[1];
    Real l_inf_g, l2_g;
    l_inf_g = pm->reduce(Parallel::ReduceMax, l_inf);
    l2_g = pm->reduce(Parallel::ReduceSum, l2);
    m_error_vals[0] = math::sqrt(l2_g);
    m_error_vals[1] = l_inf_g;
  }
  else
    m_error_vals[0] = math::sqrt(m_error_vals[0]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// passing the error committed per cell to the refinement flag
// This method could be designed in different ways:
// 1- current implementation: the user performs the transformation themselves
// in this case, they modify the itemInternal object by setting the refinement flag
// 2- the user performs the transformation themselves but stores and returns an array of flags
// the MeshRefinement class, in this case, implements a setter from the array returned here
// 3- to avoid copying the flags array, implement the converter directly in meshRefinement
// and the user only provides the error array
void ErrorEstimate::
errorToFlagConverter(RealArray& error_per_cell, const Real& refine_frac,
                     const Real& coarsen_frac, const Integer& max_level, IMesh* mesh)
{

  // Check for valid fractions..
  // The fraction values must be in [0,1]
  ARCANE_ASSERT((refine_frac >= 0. && refine_frac <= 1.), (" 0 <= refine_frac  <= 1."));
  ARCANE_ASSERT((coarsen_frac >= 0. && coarsen_frac <= 1.), ("0 <= coarsen_frac <= 1."));

  // We're getting the minimum and maximum error values
  // for the ACTIVE elements
  Real error_min = 1.e30;
  Real error_max = 0.;

  // We need to loop over all active elements to find the minimum
  IParallelMng* pm = mesh->parallelMng();
  Integer i = 0;
  ENUMERATE_CELL (icell, mesh->ownActiveCells()) { // active cells
    Cell cell = *icell;
    const Integer id = cell.localId();
    if (id >= error_per_cell.size())
      ARCANE_FATAL("Bad local_id '{0}' (max_valid={1})", id, error_per_cell.size());

    error_max = math::max(error_max, error_per_cell[i]);
    error_min = math::min(error_min, error_per_cell[i]);
    i++;
  }
  if (pm->commSize() > 1) {
    const Real error_max_g = pm->reduce(Parallel::ReduceMax, error_max);
    const Real error_min_g = pm->reduce(Parallel::ReduceMin, error_min);
    error_max = error_max_g;
    error_min = error_min_g;
  }

  // Compute the cutoff values for coarsening and refinement
  const Real error_delta = (error_max - error_min);
  //const Real parent_error_delta = parent_error_max - parent_error_min;

  const Real refine_cutoff = (1. - refine_frac) * error_max;
  const Real coarsen_cutoff = coarsen_frac * error_delta + error_min;

  //   // Print information about the error
  //   debug() << " Error Information: \n"                     <<
  // 	    << " ------------------\n"                     <<
  // 	    << "   min:              " << error_min      << "\n"
  // 	    << "   max:              " << error_max      << "\n"
  // 	    << "   delta:            " << error_delta    << "\n"
  // 	    << "     refine_cutoff:  " << refine_cutoff  << "\n"
  // 	    << "     coarsen_cutoff: " << coarsen_cutoff << "\n";

  // Tag the meshes for adaptation
  i = 0;
  ENUMERATE_CELL (icell, mesh->ownActiveCells()) { // active cells
    Cell cell = *icell;
    const Integer id = cell.localId();
    ARCANE_ASSERT((id < error_per_cell.size()), ("cell_lid < error_per_cell.size()"));

    const Real cell_error = error_per_cell[i++];

    // Flag for coarsening if error <= coarsen_fraction*delta + error_min
    if (cell_error <= coarsen_cutoff && cell.level() > 0) {
      if (cell.type() == IT_Hexaedron8) {
        cell.mutableItemBase().addFlags(ItemInternal::II_Coarsen);
      }
    }

    // Flag for refinement if error >= refinement_cutoff.
    if (cell_error >= refine_cutoff && cell.level() < max_level)
      if (cell.type() == IT_Hexaedron8) {
        cell.mutableItemBase().addFlags(ItemInternal::II_Refine);
      }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
