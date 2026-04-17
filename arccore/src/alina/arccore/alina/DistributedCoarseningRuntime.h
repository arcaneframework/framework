// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedCoarseningRuntime.h                              (C) 2000-2026 */
/*                                                                           */
/* Runtime wrapper for distributed coarsening schemes.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTEDCOARSENINGRUNTIME_H
#define ARCCORE_ALINA_MPI_DISTRIBUTEDCOARSENINGRUNTIME_H
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

#include "arccore/alina/DistributedMatrix.h"
#include "arccore/alina/DistributedCoarsening.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

enum class eDistributedCoarseningType
{
  aggregation,
  smoothed_aggregation
};

inline std::ostream& operator<<(std::ostream& os, eDistributedCoarseningType s)
{
  switch (s) {
  case eDistributedCoarseningType::aggregation:
    return os << "aggregation";
  case eDistributedCoarseningType::smoothed_aggregation:
    return os << "smoothed_aggregation";
  default:
    return os << "???";
  }
}

inline std::istream& operator>>(std::istream& in, eDistributedCoarseningType& s)
{
  std::string val;
  in >> val;

  if (val == "aggregation")
    s = eDistributedCoarseningType::aggregation;
  else if (val == "smoothed_aggregation")
    s = eDistributedCoarseningType::smoothed_aggregation;
  else
    throw std::invalid_argument("Invalid coarsening value. Valid choices are: "
                                "aggregation, smoothed_aggregation.");

  return in;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend>
struct DistributedCoarseningRuntime
{
  typedef DistributedMatrix<Backend> matrix;
  typedef PropertyTree params;

  eDistributedCoarseningType c;
  void* handle = nullptr;

  explicit DistributedCoarseningRuntime(params prm = params())
  : c(prm.get("type", eDistributedCoarseningType::smoothed_aggregation))
  {
    if (!prm.erase("type"))
      ARCCORE_ALINA_PARAM_MISSING("type");

    switch (c) {
    case eDistributedCoarseningType::aggregation: {
      typedef DistributedAggregationCoarsening<Backend> C;
      handle = static_cast<void*>(new C(prm));
    } break;
    case eDistributedCoarseningType::smoothed_aggregation: {
      typedef DistributedSmoothedAggregationCoarsening<Backend> C;
      handle = static_cast<void*>(new C(prm));
    } break;
    default:
      throw std::invalid_argument("Unsupported coarsening type");
    }
  }

  ~DistributedCoarseningRuntime()
  {
    switch (c) {
    case eDistributedCoarseningType::aggregation: {
      typedef DistributedAggregationCoarsening<Backend> C;
      delete static_cast<C*>(handle);
    } break;
    case eDistributedCoarseningType::smoothed_aggregation: {
      typedef DistributedSmoothedAggregationCoarsening<Backend> C;
      delete static_cast<C*>(handle);
    } break;
    default:
      break;
    }
  }

  std::tuple<std::shared_ptr<matrix>, std::shared_ptr<matrix>>
  transfer_operators(const matrix& A)
  {
    switch (c) {
    case eDistributedCoarseningType::aggregation: {
      typedef DistributedAggregationCoarsening<Backend> C;
      return static_cast<C*>(handle)->transfer_operators(A);
    }
    case eDistributedCoarseningType::smoothed_aggregation: {
      typedef DistributedSmoothedAggregationCoarsening<Backend> C;
      return static_cast<C*>(handle)->transfer_operators(A);
    }
    default:
      throw std::invalid_argument("Unsupported partition type");
    }
  }

  std::shared_ptr<matrix>
  coarse_operator(const matrix& A, const matrix& P, const matrix& R) const
  {
    switch (c) {
    case eDistributedCoarseningType::aggregation: {
      typedef DistributedAggregationCoarsening<Backend> C;
      return static_cast<C*>(handle)->coarse_operator(A, P, R);
    }
    case eDistributedCoarseningType::smoothed_aggregation: {
      typedef DistributedSmoothedAggregationCoarsening<Backend> C;
      return static_cast<C*>(handle)->coarse_operator(A, P, R);
    }
    default:
      throw std::invalid_argument("Unsupported partition type");
    }
  }
  friend std::ostream& operator<<(std::ostream& os, const DistributedCoarseningRuntime& w)
  {
    os << "Coarsening: " << w.c << "\n";
    return os;
  }
};

template <class Backend>
unsigned block_size(const DistributedCoarseningRuntime<Backend>& w)
{
  switch (w.c) {
  case eDistributedCoarseningType::aggregation: {
    typedef DistributedAggregationCoarsening<Backend> C;
    return block_size(*static_cast<const C*>(w.handle));
  }
  case eDistributedCoarseningType::smoothed_aggregation: {
    typedef DistributedSmoothedAggregationCoarsening<Backend> C;
    return block_size(*static_cast<const C*>(w.handle));
  }
  default:
    throw std::invalid_argument("Unsupported coarsening type");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
