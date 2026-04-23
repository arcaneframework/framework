// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlinaGlobal.h                                               (C) 2000-2026 */
/*                                                                           */
/* Déclarations générales de la composante 'arcane_alina'.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_ALINAGLOBAL_H
#define ARCCORE_ALINA_ALINAGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_alina
#define ARCCORE_ALINA_EXPORT ARCANE_EXPORT
#else
#define ARCCORE_ALINA_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Always activate profiling at the moment
#define ARCCORE_ALINA_PROFILING

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace detail
{
// Backend with scalar value_type of highest precision.
template <class B1, class B2, class Enable = void>
struct common_scalar_backend;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ExecutionContext;
class PropertyTree;
using AlinaDefaultColumnType = Int32;
using AlinaDefaultRowIndexType = Int32;
template <typename ValueType_ = double,
          typename ColumnType_ = AlinaDefaultColumnType,
          typename RowIndexType_ = AlinaDefaultRowIndexType>
class CSRMatrixView;

template <typename val_t = double,
          typename col_t = AlinaDefaultColumnType,
          typename ptr_t = AlinaDefaultRowIndexType>
struct CSRMatrix;
template <typename V, typename C, typename P>
struct BlockCSRMatrix;

template <typename IndexType_ = Int32>
class CSRRow;
template <typename IndexType_ = Int32>
class CSRRowColumnIndex;
template <typename IndexType_ = Int32>
class CSRRowColumnIterator;
template <typename IndexType_ = Int32>
class CSRRowRangeIterator;
template <typename IndexType_ = Int32>
class CSRRowRange;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
