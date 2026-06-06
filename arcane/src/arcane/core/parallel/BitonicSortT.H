// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BitonicSortT.H                                              (C) 2000-2025 */
/*                                                                           */
/* Parallel bitonic sort algorithm.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLEL_BITONICSORT_IMPL_H
#define ARCANE_CORE_PARALLEL_BITONICSORT_IMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/parallel/BitonicSort.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename KeyType, typename KeyTypeTraits> BitonicSort<KeyType, KeyTypeTraits>::
BitonicSort(IParallelMng* parallel_mng)
: TraceAccessor(parallel_mng->traceMng())
, m_parallel_mng(parallel_mng)
{
}

template <typename KeyType, typename KeyTypeTraits> BitonicSort<KeyType, KeyTypeTraits>::
BitonicSort(IParallelMng* parallel_mng, const KeyTypeTraits& traits)
: TraceAccessor(parallel_mng->traceMng())
, m_parallel_mng(parallel_mng)
, m_traits(traits)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename KeyType, typename KeyTypeTraits> void BitonicSort<KeyType, KeyTypeTraits>::
_init(ConstArrayView<KeyType> keys)
{
  m_init_size = keys.size();

  // It is essential that all ranks have the same value
  // of m_size
  m_size = m_parallel_mng->reduce(Parallel::ReduceMax, m_init_size);

  m_keys.resize(m_size);
  m_key_ranks.resize(m_size);
  m_key_indexes.resize(m_size);

  Int32 rank = m_parallel_mng->commRank();

  // The values of the tab grid variable are stored
  // in the m_key array
  for (Integer i = 0; i < m_init_size; ++i) {
    m_keys[i] = keys[i];
    m_key_indexes[i] = i;
    m_key_ranks[i] = rank;
  }

  // Fills the end of the array with non-significant values
  // So that each processor has the same number of elements
  KeyType max_value = m_traits.maxValue();
  for (Int64 i = m_init_size; i < m_size; i++) {
    m_keys[i] = max_value;
    m_key_indexes[i] = -1;
    m_key_ranks[i] = rank;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \brief Bitonic sort of the variable \a m_key

  Upon exiting the call, the key variable is entirely sorted
  in ascending order: the first elements are located
  in ascending order on processor 0, the following on 
  processor 1, etc...
  The sorting is done 'in place', so that the size
  of the arrays on the different processors is not modified.

*/
template <typename KeyType, typename KeyTypeTraits> void
BitonicSort<KeyType, KeyTypeTraits>::
sort(ConstArrayView<KeyType> keys)
{
  _init(keys);

  m_nb_merge = 0;

  info() << "BITONIC_SORT (64bit) want_rank?=" << m_want_index_and_rank
         << " size=" << m_size
         << " memsize=" << sizeof(KeyType) * m_size
         << " structsize=" << sizeof(KeyType);

  // Call to local key sort
  _localHeapSort();

  // Definition of the number of levels for the algorithm
  Int32 nb_level = 0;
  Int32 test = 1;
  Int32 nb_rank = m_parallel_mng->commSize();
  while (nb_rank > test) {
    nb_level++;
    test *= 2;
  }
  Int32 size = 1;

  // Loop over the depth levels of the algorithm
  for (Int32 ilevel = 0; ilevel < nb_level; ++ilevel) {
    size *= 2;

    // Loop over the groups of processors to be processed
    for (Int32 jproc = 0; jproc < nb_rank; jproc += size) {
      _mergeLevels(jproc, size);
    }
  }

  // The number of valid elements in the list may be less
  // than the number of allocated elements. As soon as a value is invalid,
  // we stop the list.
  // TODO: start from the end
  for (Integer i = 0; i < m_size; ++i) {
    if (!m_traits.isValid(m_keys[i])) {
      m_size = i;
      break;
    }
  }

  info() << "END_BITONIC_SORT nb_merge=" << m_nb_merge;

  m_keys.resize(m_size);
  m_key_ranks.resize(m_size);
  m_key_indexes.resize(m_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \brief Bitonic sort of the variable \a key.

  Upon exiting the call, the key variable is entirely sorted
  in ascending order: the first elements are located
  in ascending order on processor 0, the following on 
  processor 1, etc...
  The sorting is done 'in place', so that the size
  of the arrays on the different processors is not modified.
*/
template <typename KeyType, typename KeyTypeTraits> void
BitonicSort<KeyType, KeyTypeTraits>::
_mergeLevels(Int32 begin, Int32 size)
{
  // Separator stage
  for (Int32 i = 0; i < size / 2; ++i)
    _mergeProcessors(begin + i, begin + size - 1 - i);

  if (size <= 2)
    return;

  // Application of the bitonic sorter of level n/2
  Int32 div_size = size;
  while (div_size > 1) {
    div_size /= 2;
    for (Int32 iproc = 0; iproc < size; iproc += div_size)
      _separator(begin + iproc, div_size);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Bitonic sort algorithm separator stage
 */
template <typename KeyType, typename KeyTypeTraits> void BitonicSort<KeyType, KeyTypeTraits>::
_separator(Int32 begin, Int32 size)
{
  for (Int32 i = 0; i < size / 2; ++i)
    _mergeProcessors(begin + i, begin + i + size / 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Merging lists from two processors.
 *
 \param iproc1 processor with lower ID whose list is merged
 \param iproc2 processor with higher ID whose list is merged

 * This operation is the elementary building block of 
 * the bitonic merge sort algorithm. On input, it
 * takes the numbers of two processors with iproc1<iproc2.
 * On output, the m_key arrays of these two processors
 * are sorted: the smallest elements are sorted in ascending order
 * on iproc1, the largest elements are sorted in ascending order
 * on iproc2.
 */
template <typename KeyType, typename KeyTypeTraits> void BitonicSort<KeyType, KeyTypeTraits>::
_mergeProcessors(Int32 iproc1, Int32 iproc2)
{
  if (iproc1 >= iproc2)
    ARCANE_FATAL("Invalid merge iproc1={0} iproc2={1}", iproc1, iproc2);

  Int32 my_rank = m_parallel_mng->commRank();

  if (iproc2 >= m_parallel_mng->commSize())
    return;

  bool is_proc2 = (iproc2 == my_rank);
  bool is_proc1 = (iproc1 == my_rank);

  if (!is_proc1 && !is_proc2)
    return;

  info() << "SORT iproc1=" << iproc1 << " iproc2=" << iproc2;
  ++m_nb_merge;

  const Int64 buf_size = m_size;
  const bool is_want_index_rank = m_want_index_and_rank;
  ////////////////////////////////////////////
  //Processor 2 processing
  ////////////////////////////////////////////

  if (is_proc2) {
    // Send the list to proc 1
    UniqueArray<KeyType> send_keys(m_keys);
    UniqueArray<Request> requests;
    requests.add(m_traits.send(m_parallel_mng, iproc1, send_keys));
    if (is_want_index_rank) {
      requests.add(m_parallel_mng->send(m_key_indexes, iproc1, false));
      requests.add(m_parallel_mng->send(m_key_ranks, iproc1, false));
    }
    // Receive the end of the merged list by proc1
    requests.add(m_traits.recv(m_parallel_mng, iproc1, m_keys));
    if (is_want_index_rank) {
      requests.add(m_parallel_mng->recv(m_key_indexes, iproc1, false));
      requests.add(m_parallel_mng->recv(m_key_ranks, iproc1, false));
    }
    m_parallel_mng->waitAllRequests(requests);
  }

  //////////////////////////////////////
  //Processor 1 processing
  //////////////////////////////////////
  if (is_proc1) {
    // Receive the list from proc2

    UniqueArray<KeyType> buf2(buf_size);
    Int32UniqueArray buf2_proc;
    Int32UniqueArray buf2_local_id;
    // We allocate buf2 to size+1 in order to place a sentinel in
    // buf2[size]
    buf2.resize(buf_size);
    if (is_want_index_rank) {
      buf2_proc.resize(buf_size);
      buf2_local_id.resize(buf_size);
    }

    Request recv_request = m_traits.recv(m_parallel_mng, iproc2, buf2);
    m_parallel_mng->waitAllRequests(ArrayView<Request>(1, &recv_request));
    if (is_want_index_rank) {
      m_parallel_mng->recv(buf2_local_id, iproc2);
      m_parallel_mng->recv(buf2_proc, iproc2);
    }

    // to be replaced by the largest number representable by an Integer
    buf2.add(m_traits.maxValue());

    // allocation of a working buffer for merging the two lists
    Int64 total_size = m_size + buf_size;
    UniqueArray<KeyType> buf_merge(total_size);
    UniqueArray<Int32> buf_proc_merge;
    UniqueArray<Int32> buf_local_id_merge;
    if (is_want_index_rank) {
      buf_proc_merge.resize(total_size);
      buf_local_id_merge.resize(total_size);
    }

#if 0
    info() << "SIZE=" << buf_size << " total=" << total_size << " m_size=" << m_size
           << " proc2=" << iproc2 << " init-size=" << m_init_size;
#endif

    // Merging the two lists based on the assumption that each one is
    // already sorted.
    Int64 cursor1 = 0;
    Int64 cursor2 = 0;
    for (Int64 i = 0; i < total_size; ++i) {
      if (cursor1 >= m_size || (m_traits.compareLess(buf2[cursor2], m_keys[cursor1]) && cursor2 < buf_size)) {
        buf_merge[i] = buf2[cursor2];
        if (is_want_index_rank) {
          buf_local_id_merge[i] = buf2_local_id[cursor2];
          buf_proc_merge[i] = buf2_proc[cursor2];
        }
        ++cursor2;
      }
      else {
        buf_merge[i] = m_keys[cursor1];
        if (is_want_index_rank) {
          buf_local_id_merge[i] = m_key_indexes[cursor1];
          buf_proc_merge[i] = m_key_ranks[cursor1];
        }
        ++cursor1;
      }
    }
    buf2.resize(buf_size);

    // Copy and send the end of the list to proc2
    for (Integer ii = 0; ii < buf_size; ii++) {
      buf2[ii] = buf_merge[m_size + ii];
      if (is_want_index_rank) {
        buf2_local_id[ii] = buf_local_id_merge[m_size + ii];
        buf2_proc[ii] = buf_proc_merge[m_size + ii];
      }
    }
    Request send_request = m_traits.send(m_parallel_mng, iproc2, buf2);
    m_parallel_mng->waitAllRequests(ArrayView<Request>(1, &send_request));
    if (is_want_index_rank) {
      m_parallel_mng->send(buf2_local_id, iproc2);
      m_parallel_mng->send(buf2_proc, iproc2);
    }

    // Copy from the beginning of the list into the m_work_unique_id variable
    for (Integer i = 0; i < buf_size; i++) {
      m_keys[i] = buf_merge[i];
      if (is_want_index_rank) {
        m_key_indexes[i] = buf_local_id_merge[i];
        m_key_ranks[i] = buf_proc_merge[i];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Local heap sort of the variable \a m_key.
 *
 * Upon exiting the call, the variable is sorted
 * in ascending order within each processor.
 */
template <typename KeyType, typename KeyTypeTraits>
void BitonicSort<KeyType, KeyTypeTraits>::
_localHeapSort()
{
  // Special case of a list reduced to zero or one element
  if (m_size < 2)
    return;

  // Following the parent of the last element
  Int64 l = m_size / 2;
  Int64 ir = m_size - 1;
  KeyType rra;
  Int64 i = 0;
  Int64 j = 0;
  Integer tmp_local_id = NULL_ITEM_LOCAL_ID;
  Integer tmp_proc = A_NULL_RANK;

  for (;;) {
    if (l > 0) {
      --l;
      rra = m_keys[l];
      tmp_local_id = m_key_indexes[l];
      tmp_proc = m_key_ranks[l];
    }
    else {
      rra = m_keys[ir];
      tmp_local_id = m_key_indexes[ir];
      tmp_proc = m_key_ranks[ir];

      m_keys[ir] = m_keys[0];
      m_key_indexes[ir] = m_key_indexes[0];
      m_key_ranks[ir] = m_key_ranks[0];

      if (--ir == 0) {
        m_keys[0] = rra;
        m_key_indexes[0] = tmp_local_id;
        m_key_ranks[0] = tmp_proc;
        break;
      }
    }
    i = l;
    j = l + l + 1;
    while (j <= ir) {
      if (j < ir && m_traits.compareLess(m_keys[j], m_keys[j + 1]))
        ++j;
      if (m_traits.compareLess(rra, m_keys[j])) {
        m_keys[i] = m_keys[j];
        m_key_indexes[i] = m_key_indexes[j];
        m_key_ranks[i] = m_key_ranks[j];
        i = j;
        // left child
        j = j * 2 + 1;
      }
      else
        j = ir + 1;
    }
    m_keys[i] = rra;
    m_key_indexes[i] = tmp_local_id;
    m_key_ranks[i] = tmp_proc;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
