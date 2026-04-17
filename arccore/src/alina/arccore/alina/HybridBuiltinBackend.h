// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridBuiltinBackend.h                                      (C) 2000-2026 */
/*                                                                           */
/* Builtin backend that uses scalar and block format matrix.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_HYBRIDBUILTINBACKEND_H
#define ARCCORE_ALINA_HYBRIDBUILTINBACKEND_H
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
#include "arccore/alina/Adapters.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Hybrid backend uses scalar matrices to build the hierarchy,
 * but stores the computed matrices in the block format.
 */
template <typename BlockType, typename ColumnType = ptrdiff_t, typename PointerType = ColumnType>
struct HybridBuiltinBackend : public BuiltinBackend<typename math::scalar_of<BlockType>::type, ColumnType, PointerType>
{
  typedef typename math::scalar_of<BlockType>::type ScalarType;
  typedef BuiltinBackend<ScalarType, ColumnType, PointerType> Base;
  typedef CSRMatrix<BlockType, ColumnType, PointerType> matrix;
  struct provides_row_iterator : std::false_type
  {};

  static std::shared_ptr<matrix>
  copy_matrix(std::shared_ptr<typename Base::matrix> As, const typename Base::params&)
  {
    return std::make_shared<matrix>(Alina::adapter::block_matrix<BlockType>(*As));
  }
};

template <typename B1, typename B2, typename C, typename P>
struct backends_compatible<HybridBuiltinBackend<B1, C, P>, HybridBuiltinBackend<B2, C, P>> : std::true_type
{};

template <typename T1, typename B2, typename C, typename P>
struct backends_compatible<BuiltinBackend<T1, C, P>, HybridBuiltinBackend<B2, C, P>> : std::true_type
{};

template <typename B1, typename T2, typename C, typename P>
struct backends_compatible<HybridBuiltinBackend<B1, C, P>, BuiltinBackend<T2, C, P>> : std::true_type
{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::backend

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
