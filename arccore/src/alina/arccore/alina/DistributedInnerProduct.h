// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedInnerProduct.h                                   (C) 2000-2026 */
/*                                                                           */
/* Distributed inner products of two vectors.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTEDINNERPRODUCT_H
#define ARCCORE_ALINA_MPI_DISTRIBUTEDINNERPRODUCT_H
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

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/MessagePassingUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Inner product for distributed vectors.
 */
struct DistributedInnerProduct
{
  mpi_communicator comm;

  explicit DistributedInnerProduct(mpi_communicator comm)
  : comm(comm)
  {}

  template <class Vec1, class Vec2>
  typename math::inner_product_impl<typename backend::value_type<Vec1>::type>::return_type
  operator()(const Vec1& x, const Vec2& y) const
  {
    typedef typename backend::value_type<Vec1>::type value_type;
    typedef typename math::inner_product_impl<value_type>::return_type coef_type;

    ARCCORE_ALINA_TIC("inner product");
    coef_type sum = comm.reduceSum(backend::inner_product(x, y));
    ARCCORE_ALINA_TOC("inner product");

    return sum;
  }

  int rank() const
  {
    return comm.rank;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
