// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MTLBackEnd                                                  (C) 2000-2025 */
/*                                                                           */
/* Tools for Hypre backend                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ALIEN_KERNELS_HYPRE_HYPREBACKEND_H
#define ALIEN_KERNELS_HYPRE_HYPREBACKEND_H

#include <arccore/message_passing/MessagePassingGlobal.h>
#include <alien/core/backend/BackEnd.h>
#include <alien/AlienExternalPackagesPrecomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IOptionsHypreSolver;

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MultiVectorImpl;
class HypreMatrix;
class HypreVector;
class Space;
template <class Matrix, class Vector> class IInternalLinearAlgebra;
template <class Matrix, class Vector> class IInternalLinearSolver;

extern ALIEN_EXTERNAL_PACKAGES_EXPORT IInternalLinearAlgebra<HypreMatrix, HypreVector>*
HypreInternalLinearAlgebraFactory();

extern IInternalLinearSolver<HypreMatrix, HypreVector>* HypreInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsHypreSolver* options);

/*---------------------------------------------------------------------------*/

namespace BackEnd {
  namespace tag {
    struct hypre
    {
    };
  } // namespace tag
}

template <> struct AlgebraTraits<BackEnd::tag::hypre>
{
  typedef HypreMatrix matrix_type;
  typedef HypreVector vector_type;
  typedef IOptionsHypreSolver options_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;
  static algebra_type* algebra_factory(
      [[maybe_unused]] Arccore::MessagePassing::IMessagePassingMng* p_mng = nullptr)
  {
    return HypreInternalLinearAlgebraFactory();
  }
  static solver_type* solver_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng, options_type* options)
  {
    return HypreInternalLinearSolverFactory(p_mng, options);
  }
  static BackEndId name() { return "hypre"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_KERNELS_HYPRE_HYPREBACKEND_H */
