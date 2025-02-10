// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <arccore/message_passing/MessagePassingGlobal.h>
#include <alien/core/backend/BackEnd.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

class RedistributorLinearSolver;
class RedistributorMatrix;
class RedistributorVector;
class Space;
template <class Matrix, class Vector>
class IInternalLinearAlgebra;
template <class Matrix, class Vector>
class IInternalLinearSolver;

extern IInternalLinearAlgebra<RedistributorMatrix, RedistributorVector>*
redistributorLinearAlgebraFactory();

extern IInternalLinearSolver<RedistributorMatrix, RedistributorVector>*
redistributorLinearSolverFactory(IMessagePassingMng* p_mng);

/*---------------------------------------------------------------------------*/

namespace BackEnd
{
  namespace tag
  {
    struct redistributor
    {};
  } // namespace tag
} // namespace BackEnd

template <>
struct AlgebraTraits<BackEnd::tag::redistributor>
{
  typedef RedistributorMatrix matrix_type;
  typedef RedistributorVector vector_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;
  static algebra_type* algebra_factory() { return redistributorLinearAlgebraFactory(); }
  static solver_type* solver_factory(IMessagePassingMng* p_mng)
  {
    return redistributorLinearSolverFactory(p_mng);
  }
  static BackEndId name() { return "redistributor"; }
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
