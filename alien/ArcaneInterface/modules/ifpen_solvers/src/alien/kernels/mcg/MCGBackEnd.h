// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "alien/utils/Precomp.h"
#include "alien/core/backend/BackEnd.h"
#include "alien/kernels/mcg/data_structure/MemoryDomain.h"

class IOptionsMCGSolver;

namespace Alien {

class MultiVectorImpl;
template<typename NumT,MCGInternal::eMemoryDomain Domain>
class MCGMatrix;
template<typename NumT,MCGInternal::eMemoryDomain Domain>
class MCGVector;
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
    struct mcgsolver_gpu
    {};
  }
}

template <> struct AlgebraTraits<BackEnd::tag::mcgsolver>
{
  using matrix_type = MCGMatrix<Real,MCGInternal::eMemoryDomain::Host>;
  using vector_type = MCGVector<Real,MCGInternal::eMemoryDomain::Host>;

  using options_type = IOptionsMCGSolver;
  using algebra_type = ILinearAlgebra;
  using solver_type = ILinearSolver;

  static algebra_type* algebra_factory() { return MCGInternalLinearAlgebraFactory(); }

  static solver_type* solver_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng, options_type* options)
  {
    return MCGInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "mcgsolver"; }
};

template <> struct AlgebraTraits<BackEnd::tag::mcgsolver_gpu>
{
  using matrix_type = MCGMatrix<Real,MCGInternal::eMemoryDomain::Device>;
  using vector_type = MCGVector<Real,MCGInternal::eMemoryDomain::Device>;

  using options_type = IOptionsMCGSolver;
  using algebra_type = ILinearAlgebra;
  using solver_type = ILinearSolver;

  static algebra_type* algebra_factory() { return MCGInternalLinearAlgebraFactory(); }

  static solver_type* solver_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng, options_type* options)
  {
    return MCGInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "mcgsolver_gpu"; }
};
} // namespace Alien
