// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <alien/utils/Precomp.h>
#include <alien/core/backend/BackEnd.h>

class IOptionsMCGSolver;

namespace Alien {

class MultiVectorImpl;
class MCGMatrix;
class MCGVector;
class MCGCompositeMatrix;
class MCGCompositeVector;
class MatrixData;
class MatrixExp;
class VectorData;
class VectorData;
class VectorExp;
class ILinearSolver;
class ILinearAlgebra;

extern ILinearAlgebra* MCGInternalLinearAlgebraFactory();

extern ILinearSolver* MCGInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsMCGSolver* options);

/*---------------------------------------------------------------------------*/

namespace BackEnd {
  namespace tag {
    struct mcgsolver
    {};
    struct mcgsolver_composite
    {};
  }
}

template <> struct AlgebraTraits<BackEnd::tag::mcgsolver>
{
  typedef MCGMatrix matrix_type;
  typedef MCGVector vector_type;

  typedef IOptionsMCGSolver options_type;
  typedef ILinearAlgebra algebra_type;
  typedef ILinearSolver solver_type;

  static algebra_type* algebra_factory() { return MCGInternalLinearAlgebraFactory(); }

  static solver_type* solver_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng, options_type* options)
  {
    return MCGInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "mcgsolver"; }
};

template <> struct AlgebraTraits<BackEnd::tag::mcgsolver_composite>
{
  typedef MCGCompositeMatrix matrix_type;
  typedef MCGCompositeVector vector_type;

  typedef IOptionsMCGSolver options_type;
  typedef ILinearAlgebra algebra_type;
  typedef ILinearSolver solver_type;

  static algebra_type* algebra_factory() { return MCGInternalLinearAlgebraFactory(); }

  static solver_type* solver_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng, options_type* options)
  {
    return MCGInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "mcgsolver_composite"; }
};
} // namespace Alien
