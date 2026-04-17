// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SolverRuntime.h                                             (C) 2000-2026 */
/*                                                                           */
/* Runtime-configurable solvers.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_SOLVERRUNTIME_H
#define ARCCORE_ALINA_SOLVERRUNTIME_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/ConjugateGradientSolver.h"
#include "arccore/alina/BiCGStabSolver.h"
#include "arccore/alina/BiCGStabLSolver.h"
#include "arccore/alina/GMRESSolver.h"
#include "arccore/alina/LooseGMRESSolver.h"
#include "arccore/alina/FlexibleGMRESSolver.h"
#include "arccore/alina/IDRSSolver.h"
#include "arccore/alina/RichardsonSolver.h"
#include "arccore/alina/PreconditionerOnlySolver.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum class eSolverType
{
  cg, ///< Conjugate gradients method
  ConjugateGradientSolver = cg, ///< Conjugate gradients method
  bicgstab, ///< BiConjugate Gradient Stabilized
  BiCGStabSolver = bicgstab, ///< BiConjugate Gradient Stabilized
  bicgstabl, ///< BiCGStab(ell)
  BiCGStabLSolver = bicgstabl, ///< BiCGStab(ell)
  gmres, ///< GMRES
  GMRESSolver = gmres, ///< GMRES
  lgmres, ///< LGMRES
  LooseGMRESSolver = lgmres, ///< LGMRES
  fgmres, ///< FGMRES
  FlexibleGMRESSolver = fgmres,
  idrs, ///< IDR(s)
  IDRSSolver = idrs, ///< IDR(s)
  richardson, ///< Richardson iteration
  RichardsonSolver = richardson, ///< Richardson iteration
  preonly, ///< Only apply preconditioner once
  PreconditionerOnlySolver = preonly
};

inline std::ostream& operator<<(std::ostream& os, eSolverType s)
{
  switch (s) {
  case eSolverType::cg:
    return os << "cg";
  case eSolverType::bicgstab:
    return os << "bicgstab";
  case eSolverType::bicgstabl:
    return os << "bicgstabl";
  case eSolverType::gmres:
    return os << "gmres";
  case eSolverType::lgmres:
    return os << "lgmres";
  case eSolverType::fgmres:
    return os << "fgmres";
  case eSolverType::idrs:
    return os << "idrs";
  case eSolverType::richardson:
    return os << "richardson";
  case eSolverType::preonly:
    return os << "preonly";
  default:
    return os << "???";
  }
}

inline std::istream& operator>>(std::istream& in, eSolverType& s)
{
  std::string val;
  in >> val;

  if (val == "cg")
    s = eSolverType::cg;
  else if (val == "bicgstab")
    s = eSolverType::bicgstab;
  else if (val == "bicgstabl")
    s = eSolverType::bicgstabl;
  else if (val == "gmres")
    s = eSolverType::gmres;
  else if (val == "lgmres")
    s = eSolverType::lgmres;
  else if (val == "fgmres")
    s = eSolverType::fgmres;
  else if (val == "idrs")
    s = eSolverType::idrs;
  else if (val == "richardson")
    s = eSolverType::richardson;
  else if (val == "preonly")
    s = eSolverType::preonly;
  else
    throw std::invalid_argument("Invalid solver value. Valid choices are: "
                                "cg, bicgstab, bicgstabl, gmres, lgmres, fgmres, idrs, richardson, preonly.");

  return in;
}

#define ARCCORE_ALINA_ALL_RUNTIME_SOLVER() \
  ARCCORE_ALINA_RUNTIME_SOLVER(ConjugateGradientSolver); \
  ARCCORE_ALINA_RUNTIME_SOLVER(BiCGStabSolver); \
  ARCCORE_ALINA_RUNTIME_SOLVER(BiCGStabLSolver); \
  ARCCORE_ALINA_RUNTIME_SOLVER(GMRESSolver); \
  ARCCORE_ALINA_RUNTIME_SOLVER(LooseGMRESSolver); \
  ARCCORE_ALINA_RUNTIME_SOLVER(FlexibleGMRESSolver); \
  ARCCORE_ALINA_RUNTIME_SOLVER(IDRSSolver); \
  ARCCORE_ALINA_RUNTIME_SOLVER(RichardsonSolver); \
  ARCCORE_ALINA_RUNTIME_SOLVER(PreconditionerOnlySolver)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Runtime-configurable wrappers around iterative solvers.
 */
template <class Backend, class InnerProduct = detail::default_inner_product>
struct SolverRuntime
{
  typedef PropertyTree params;
  typedef typename Backend::params backend_params;
  typedef typename Backend::value_type value_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef Backend backend_type;
  using BackendType = backend_type;

  eSolverType m_solver_type;
  SolverBase* m_solver = nullptr;

  explicit SolverRuntime(size_t n, params prm = params(),
                         const backend_params& bprm = backend_params(),
                         const InnerProduct& inner_product = InnerProduct())
  : m_solver_type(prm.get("type", eSolverType::bicgstab))
  {
    if (!prm.erase("type"))
      ARCCORE_ALINA_PARAM_MISSING("type");

    switch (m_solver_type) {

#define ARCCORE_ALINA_RUNTIME_SOLVER(type) \
  case eSolverType::type: \
    m_solver = new type<Backend, InnerProduct>(n, prm, bprm, inner_product); \
    break

      ARCCORE_ALINA_ALL_RUNTIME_SOLVER();

#undef ARCCORE_ALINA_RUNTIME_SOLVER

    default:
      ARCCORE_FATAL("Unsupported solver type type={0}", m_solver_type);
    }
  }

  ~SolverRuntime()
  {
    delete m_solver;
  }

  template <class Matrix, class Precond, class Vec1, class Vec2>
  SolverResult operator()(const Matrix& A, const Precond& P, const Vec1& rhs, Vec2&& x) const
  {
    switch (m_solver_type) {

#define ARCCORE_ALINA_RUNTIME_SOLVER(type) \
  case eSolverType::type: \
    return static_cast<type<Backend, InnerProduct>*>(m_solver)->operator()(A, P, rhs, x)

      ARCCORE_ALINA_ALL_RUNTIME_SOLVER();

#undef ARCCORE_ALINA_RUNTIME_SOLVER

    default:
      ARCCORE_FATAL("Unsupported solver type type={0}", m_solver_type);
    }
  }

  template <class Precond, class Vec1, class Vec2>
  SolverResult operator()(const Precond& P, const Vec1& rhs, Vec2&& x) const
  {
    return (*this)(P.system_matrix(), P, rhs, x);
  }

  friend std::ostream& operator<<(std::ostream& os, const SolverRuntime& w)
  {
    switch (w.m_solver_type) {

#define ARCCORE_ALINA_RUNTIME_SOLVER(type) \
  case eSolverType::type: \
    return os << *static_cast<type<Backend, InnerProduct>*>(w.m_solver)

      ARCCORE_ALINA_ALL_RUNTIME_SOLVER();

#undef ARCCORE_ALINA_RUNTIME_SOLVER

    default:
      ARCCORE_FATAL("Unsupported solver type type={0}", w.m_solver_type);
    }
  }

  size_t bytes() const
  {
    return m_solver->bytes();
  }
};

#undef ARCCORE_ALINA_ALL_RUNTIME_SOLVER

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
