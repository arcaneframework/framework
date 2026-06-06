// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelSort.h                                             (C) 2000-2024 */
/*                                                                           */
/* Interface of a parallel sorting algorithm.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELSORT_H
#define ARCANE_CORE_IPARALLELSORT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a parallel sorting algorithm
 *
 * The key type must be comparable and possess the operator operator<.
 *
 * For now, this interface is implemented for the following types:
 * Int32, Int64, and Real.
 *
 * The sort() method performs the sorting. After sorting, it is possible
 * to retrieve the rank and the index of its origin for each key,
 * via keyRanks() and keyIndexes(). The sorted keys are accessible
 * via keys().
 */
template <typename KeyType>
class IParallelSort
{
 public:

  virtual ~IParallelSort() = default;

 public:

  /*!
   * \brief Parallel sorting of the keys \a keys.
   *
   * This method is collective.
   * The sort is global, with each rank receiving its list of keys \a keys.
   */
  virtual void sort(ConstArrayView<KeyType> keys) = 0;

  //! Array of keys
  virtual ConstArrayView<KeyType> keys() const = 0;

  //! Array of ranks of the original processor containing the key
  virtual Int32ConstArrayView keyRanks() const = 0;

  //! Array of indices of the key in the original processor.
  virtual Int32ConstArrayView keyIndexes() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
