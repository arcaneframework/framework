// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BitonicSort.h                                               (C) 2000-2025 */
/*                                                                           */
/* Parallel bitonic sort algorithm.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLEL_BITONICSORT_H
#define ARCANE_CORE_PARALLEL_BITONICSORT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/UniqueArray.h"

#include "arcane/core/IParallelSort.h"
#include "arcane/core/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Provides the necessary operations for sorting via the
 * \a BitonicSort class.
 */
template <typename KeyType>
class BitonicSortDefaultTraits
{
 public:

  static bool compareLess(const KeyType& k1, const KeyType& k2)
  {
    return k1 < k2;
  }
  static Request send(IParallelMng* pm, Int32 rank, ConstArrayView<KeyType> values)
  {
    return pm->send(values, rank, false);
  }
  static Request recv(IParallelMng* pm, Int32 rank, ArrayView<KeyType> values)
  {
    return pm->recv(values, rank, false);
  }
  //! Maximum possible value for the key.
  static KeyType maxValue()
  {
    //return ARCANE_INTEGER_MAX-1;
    return std::numeric_limits<KeyType>::max();
  }
  // Indicates if the key is valid. It must be invalid if k==maxValue()
  static bool isValid(const KeyType& k)
  {
    return k != maxValue();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Parallel bitonic sort algorithm.
 *
 * The key type can be arbitrary, but it must possess a comparison operator.
 * The necessary characteristics are provided by the
 * KeyTypeTraits class. The implementation provides operations for Int32,
 * Int64, and Real types via the
 * \a BitonicSortDefaultTraits class. For other types, it is necessary to
 * specialize this class.
 *
 * The sort() method performs the sort. After calling this method, it is
 * possible to retrieve the list of keys via \a keys() and the ranks
 * and indices in the original list of each key element via
 * the keyRanks() and keyIndexes() methods. If this information is not
 * useful, it is possible to call setNeedIndexAndRank() to disable it, which
 * allows the sort to be slightly accelerated.
 *
 * The sort is performed such that the elements are sorted in ascending order
 * starting with rank processor 0, then rank 1, and so on until the end. For
 * example, for a list of 5000 elements distributed
 * over 4 ranks, the rank 0 processor will have the 1250 smallest elements
 * at the end of the sort, the rank 1 processor the next 1250 elements, and so on.
 *
 * To accelerate the algorithm, it is preferable that all processors
 * have approximately the same number of elements in their list initially.
 * At the end of the sort, it is possible that not all processors have the same
 * number of elements in the list, and notably the highest-ranked processors
 * may not have any elements.
 */
template <typename KeyType, typename KeyTypeTraits = BitonicSortDefaultTraits<KeyType>>
class BitonicSort
: public TraceAccessor
, public IParallelSort<KeyType>
{
 public:

  explicit BitonicSort(IParallelMng* parallel_mng);
  explicit BitonicSort(IParallelMng* parallel_mng, const KeyTypeTraits& traits);

 public:

  /*!
   * \brief Parallelly sorts the elements of \a keys on all ranks.
   *
   * This operation is collective.
   */
  inline void sort(ConstArrayView<KeyType> keys) override;

  //! After a sort, returns the list of elements on this rank.
  ConstArrayView<KeyType> keys() const override { return m_keys; }

  //! After a sort, returns the array of original ranks of the elements of keys().
  Int32ConstArrayView keyRanks() const override { return m_key_ranks; }

  //! After a sort, returns the array of indices in the original list of the elements of keys().
  Int32ConstArrayView keyIndexes() const override { return m_key_indexes; }

 public:

  void setNeedIndexAndRank(bool want_index_and_rank)
  {
    m_want_index_and_rank = want_index_and_rank;
  }

 private:

  void _mergeLevels(Int32 begin, Int32 size);
  void _mergeProcessors(Int32 proc1, Int32 proc2);
  void _separator(Int32 begin, Int32 size);
  void _localHeapSort();

 private:

  //! Variable containing the sorting key
  UniqueArray<KeyType> m_keys;
  //! Array containing the rank of the processor where the key is located
  UniqueArray<Int32> m_key_ranks;
  //! Array containing the index of the key within the processor
  UniqueArray<Int32> m_key_indexes;
  //! Parallelism manager
  IParallelMng* m_parallel_mng = nullptr;
  //! Number of local elements
  Int64 m_init_size = 0;
  //! Number of local elements for the bitonic sort
  Int64 m_size = 0;

  //! Indicates whether rank and index information is desired
  bool m_want_index_and_rank = true;

  //! Statistics on the number of message levels
  Integer m_nb_merge = 0;

  KeyTypeTraits m_traits;

 private:

  void _init(ConstArrayView<KeyType> keys);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
