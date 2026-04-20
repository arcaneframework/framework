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

#include "arccore/base/NotSupportedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum class eDistributedDirectSolverType
{
  skyline_lu
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream& operator<<(std::ostream& os, eDistributedDirectSolverType s)
{
  switch (s) {
  case eDistributedDirectSolverType::skyline_lu:
    return os << "skyline_lu";
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
  else
    ARCCORE_FATAL("Invalid direct solver value. Valid choices are: 'skyline_lu'");
  return in;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Runtime wrapper for distributed direct solvers.
 */
template <typename Backend>
class DistributedDirectSolverRuntime
{
 public:

  typedef typename Backend::value_type value_type;
  typedef Alina::PropertyTree params;
  using SkylineSolverType = DistributedSkylineLUDirectSolver<Backend>;

  template <class Matrix>
  DistributedDirectSolverRuntime(Alina::mpi_communicator comm, const Matrix& A, params prm = params())
  : s(prm.get("type", eDistributedDirectSolverType::skyline_lu))
  {
    if (!prm.erase("type"))
      ARCCORE_ALINA_PARAM_MISSING("type");

    switch (s) {
    case eDistributedDirectSolverType::skyline_lu: {
      handle = static_cast<void*>(new SkylineSolverType(comm, A, prm));
    } break;
    default:
      ARCCORE_THROW(NotSupportedException, "Invalid solver type '{0}'", s);
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
      static_cast<const SkylineSolverType*>(handle)->operator()(rhs, x);
    } break;
    default:
      ARCCORE_THROW(NotSupportedException, "Invalid solver type '{0}'", s);
    }
  }

  ~DistributedDirectSolverRuntime()
  {
    switch (s) {
    case eDistributedDirectSolverType::skyline_lu: {
      delete static_cast<SkylineSolverType*>(handle);
    } break;
    default:
      break;
    }
  }

 public:

  eDistributedDirectSolverType type() const { return s; }

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
