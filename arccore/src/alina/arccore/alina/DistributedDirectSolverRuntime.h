// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedDirectSolverRuntime.h                            (C) 2000-2026 */
/*                                                                           */
/* Runtime wrapper for distributed direct solvers.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTEDDIRECTSOLVERRUNTIME_H
#define ARCCORE_ALINA_MPI_DISTRIBUTEDDIRECTSOLVERRUNTIME_H
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

#include "arccore/alina/DistributedSkylineLUDirectSolver.h"
#ifdef ARCCORE_ALINA_HAVE_EIGEN
#include "arccore/alina/DistributedEigenSparseLUDirectSolver.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum class eDistributedDirectSolverType
{
  skyline_lu
#ifdef ARCCORE_ALINA_HAVE_EIGEN
  ,
  eigen_splu
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream& operator<<(std::ostream& os, eDistributedDirectSolverType s)
{
  switch (s) {
  case eDistributedDirectSolverType::skyline_lu:
    return os << "skyline_lu";
#ifdef ARCCORE_ALINA_HAVE_EIGEN
  case eDistributedDirectSolverType::eigen_splu:
    return os << "eigen_splu";
#endif
  default:
    return os << "???";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::istream& operator>>(std::istream& in, eDistributedDirectSolverType& s)
{
  std::string val;
  in >> val;

  if (val == "skyline_lu")
    s = eDistributedDirectSolverType::skyline_lu;
#ifdef ARCCORE_ALINA_HAVE_EIGEN
  else if (val == "eigen_splu")
    s = eDistributedDirectSolverType::eigen_splu;
#endif
  else
    throw std::invalid_argument("Invalid direct solver value. Valid choices are: "
                                "skyline_lu"
#ifdef ARCCORE_ALINA_HAVE_EIGEN
                                ", eigen_splu"
#endif
                                ".");

  return in;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Runtime wrapper for distributed direct solvers.
 */
template <class value_type>
class DistributedDirectSolverRuntime
{
 public:

  typedef Alina::PropertyTree params;

  template <class Matrix>
  DistributedDirectSolverRuntime(Alina::mpi_communicator comm, const Matrix& A, params prm = params())
  : s(prm.get("type", eDistributedDirectSolverType::skyline_lu))
  {
    if (!prm.erase("type"))
      ARCCORE_ALINA_PARAM_MISSING("type");

    switch (s) {
    case eDistributedDirectSolverType::skyline_lu: {
      typedef DistributedSkylineLUDirectSolver<value_type> S;
      handle = static_cast<void*>(new S(comm, A, prm));
    } break;
#ifdef ARCCORE_ALINA_HAVE_EIGEN
    case eDistributedDirectSolverType::eigen_splu: {
      typedef DistributedEigenSparseLUDirectSolver<value_type> S;
      do_construct<S, value_type>(comm, A, prm);
    } break;
#endif
    default:
      throw std::invalid_argument("Unsupported direct solver type");
    }
  }

  static size_t coarse_enough()
  {
    return 3000 / math::static_rows<value_type>::value;
  }

  template <class Vec1, class Vec2>
  void operator()(const Vec1& rhs, Vec2& x) const
  {
    switch (s) {
    case eDistributedDirectSolverType::skyline_lu: {
      typedef DistributedSkylineLUDirectSolver<value_type> S;
      static_cast<const S*>(handle)->operator()(rhs, x);
    } break;
#ifdef ARCCORE_ALINA_HAVE_EIGEN
    case eDistributedDirectSolverType::eigen_splu: {
      typedef DistributedEigenSparseLUDirectSolver<value_type> S;
      do_solve<S, value_type>(rhs, x);
    } break;
#endif
    default:
      throw std::invalid_argument("Unsupported direct solver type");
    }
  }

  ~DistributedDirectSolverRuntime()
  {
    switch (s) {
    case eDistributedDirectSolverType::skyline_lu: {
      typedef DistributedSkylineLUDirectSolver<value_type> S;
      delete static_cast<S*>(handle);
    } break;
#ifdef ARCCORE_ALINA_HAVE_EIGEN
    case eDistributedDirectSolverType::eigen_splu: {
      typedef DistributedEigenSparseLUDirectSolver<value_type> S;
      do_destruct<S, value_type>();
    } break;
#endif
    default:
      break;
    }
  }

 private:

  eDistributedDirectSolverType s;
  void* handle = nullptr;

  template <class S, class V, class Matrix>
  typename std::enable_if<std::is_same<V, float>::value || std::is_same<V, double>::value, void>::type
  do_construct(Alina::mpi_communicator comm, const Matrix& A, const params& prm)
  {
    handle = static_cast<void*>(new S(comm, A, prm));
  }

  template <class S, class V, class Matrix>
  typename std::enable_if<!std::is_same<V, float>::value && !std::is_same<V, double>::value, void>::type
  do_construct(Alina::mpi_communicator, const Matrix&, const params&)
  {
    throw std::logic_error("The direct solver does not support the value type");
  }

  template <class S, class V, class Vec1, class Vec2>
  typename std::enable_if<std::is_same<V, float>::value || std::is_same<V, double>::value, void>::type
  do_solve(const Vec1& rhs, Vec2& x) const
  {
    static_cast<const S*>(handle)->operator()(rhs, x);
  }

  template <class S, class V, class Vec1, class Vec2>
  typename std::enable_if<!std::is_same<V, float>::value && !std::is_same<V, double>::value, void>::type
  do_solve(const Vec1&, Vec2&) const
  {
    throw std::logic_error("The direct solver does not support the value type");
  }

  template <class S, class V>
  typename std::enable_if<std::is_same<V, float>::value || std::is_same<V, double>::value, void>::type
  do_destruct()
  {
    delete static_cast<S*>(handle);
  }

  template <class S, class V>
  typename std::enable_if<!std::is_same<V, float>::value && !std::is_same<V, double>::value, void>::type
  do_destruct()
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
