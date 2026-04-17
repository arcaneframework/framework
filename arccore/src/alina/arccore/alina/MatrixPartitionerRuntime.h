// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatrixPartitionerRuntime.h                                  (C) 2000-2026 */
/*                                                                           */
/* Runtime-configurable wrapper around matrix partitioner.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MATRIXPARTITIONERRUNTIME_H
#define ARCCORE_ALINA_MATRIXPARTITIONERRUNTIME_H
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

#include <memory>

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/SimpleMatrixPartitioner.h"
#if defined(ARCCORE_ALINA_HAVE_PARMETIS)
#include "arccore/alina/ParmetisMatrixPartitioner.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum class eMatrixPartitionerType
{
  merge
#ifdef ARCCORE_ALINA_HAVE_PARMETIS
  ,
  parmetis
#endif
};

inline std::ostream&
operator<<(std::ostream& os, eMatrixPartitionerType s)
{
  switch (s) {
  case eMatrixPartitionerType::merge:
    return os << "merge";
#ifdef ARCCORE_ALINA_HAVE_PARMETIS
  case eMatrixPartitionerType::parmetis:
    return os << "parmetis";
#endif
  default:
    return os << "???";
  }
}

inline std::istream&
operator>>(std::istream& in, eMatrixPartitionerType& s)
{
  std::string val;
  in >> val;

  if (val == "merge")
    s = eMatrixPartitionerType::merge;
#ifdef ARCCORE_ALINA_HAVE_PARMETIS
  else if (val == "parmetis")
    s = eMatrixPartitionerType::parmetis;
#endif
  else
    throw std::invalid_argument("Invalid partitioner value. Valid choices are: "
                                "merge"
#ifdef ARCCORE_ALINA_HAVE_PARMETIS
                                ", parmetis"
#endif
                                ".");

  return in;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Runtime-configurable wrapper around matrix partitioner.
 */
template <class Backend>
struct MatrixPartitionerRuntime
{
  typedef DistributedMatrix<Backend> matrix;
  typedef PropertyTree params;

  eMatrixPartitionerType t;
  void* handle;

  MatrixPartitionerRuntime(params prm = params())
  : t(prm.get("type",
#if defined(ARCCORE_ALINA_HAVE_PARMETIS)
              eMatrixPartitionerType::parmetis
#else
              merge
#endif
              ))
  , handle(0)
  {
    if (!prm.erase("type"))
      ARCCORE_ALINA_PARAM_MISSING("type");

    switch (t) {
    case eMatrixPartitionerType::merge: {
      typedef SimpleMatrixPartitioner<Backend> R;
      handle = static_cast<void*>(new R(prm));
    } break;
#ifdef ARCCORE_ALINA_HAVE_PARMETIS
    case eMatrixPartitionerType::parmetis: {
      typedef ParmetisMatrixPartitioner<Backend> R;
      handle = static_cast<void*>(new R(prm));
    } break;
#endif
    default:
      throw std::invalid_argument("Unsupported partition type");
    }
  }

  ~MatrixPartitionerRuntime()
  {
    switch (t) {
    case eMatrixPartitionerType::merge: {
      typedef SimpleMatrixPartitioner<Backend> R;
      delete static_cast<R*>(handle);
    } break;
#ifdef ARCCORE_ALINA_HAVE_PARMETIS
    case eMatrixPartitionerType::parmetis: {
      typedef ParmetisMatrixPartitioner<Backend> R;
      delete static_cast<R*>(handle);
    } break;
#endif
    default:
      break;
    }
  }

  bool is_needed(const matrix& A) const
  {
    switch (t) {
    case eMatrixPartitionerType::merge: {
      typedef SimpleMatrixPartitioner<Backend> R;
      return static_cast<const R*>(handle)->is_needed(A);
    }
#ifdef ARCCORE_ALINA_HAVE_PARMETIS
    case eMatrixPartitionerType::parmetis: {
      typedef ParmetisMatrixPartitioner<Backend> R;
      return static_cast<const R*>(handle)->is_needed(A);
    }
#endif
    default:
      throw std::invalid_argument("Unsupported partition type");
    }
  }

  std::shared_ptr<matrix> operator()(const matrix& A, unsigned block_size = 1) const
  {
    switch (t) {
    case eMatrixPartitionerType::merge: {
      typedef SimpleMatrixPartitioner<Backend> R;
      return static_cast<const R*>(handle)->operator()(A, block_size);
    }
#ifdef ARCCORE_ALINA_HAVE_PARMETIS
    case eMatrixPartitionerType::parmetis: {
      typedef ParmetisMatrixPartitioner<Backend> R;
      return static_cast<const R*>(handle)->operator()(A, block_size);
    }
#endif
    default:
      throw std::invalid_argument("Unsupported partition type");
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
