// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IBufferCopier.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface for buffer copying.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_IBUFFERCOPIER_H
#define ARCANE_IMPL_IBUFFERCOPIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryView.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/GroupIndexTable.h"
#include "arcane/accelerator/core/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for copying elements between two regions with indexing.
 *
 * The copy methods can be asynchronous. You must call barrier()
 * to ensure that these copies are finished before using the
 * buffer values.
 */
class IBufferCopier
{
 public:

  virtual ~IBufferCopier() = default;

 public:

  virtual void copyFromBufferAsync(ConstArrayView<Int32> indexes,
                                   ConstMemoryView buffer,
                                   MutableMemoryView var_value) = 0;

  virtual void copyToBufferAsync(ConstArrayView<Int32> indexes,
                                 MutableMemoryView buffer,
                                 ConstMemoryView var_value) = 0;

  //! Blocks until the copies are finished.
  virtual void barrier() = 0;

 public:

  virtual void setRunQueue(const RunQueue& queue) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DirectBufferCopier
: public IBufferCopier
{
 public:

  void copyFromBufferAsync(ConstArrayView<Int32> indexes,
                           ConstMemoryView buffer,
                           MutableMemoryView var_value) override
  {
    RunQueue* q = (m_queue.isNull()) ? nullptr : &m_queue;
    MemoryUtils::copyWithIndexedSource(var_value, buffer, indexes, q);
  }

  void copyToBufferAsync(ConstArrayView<Int32> indexes,
                         MutableMemoryView buffer,
                         ConstMemoryView var_value) override
  {
    RunQueue* q = (m_queue.isNull()) ? nullptr : &m_queue;
    MemoryUtils::copyWithIndexedDestination(buffer, var_value, indexes, q);
  }

  void barrier() override;
  void setRunQueue(const RunQueue& queue) override { m_queue = queue; }

 private:

  RunQueue m_queue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TableBufferCopier
: public IBufferCopier
{
 public:

  explicit TableBufferCopier(GroupIndexTable* table)
  : m_table(table)
  {}

  void copyFromBufferAsync(ConstArrayView<Int32> indexes,
                           ConstMemoryView buffer,
                           MutableMemoryView var_value) override
  {
    UniqueArray<Int32> final_indexes(MemoryUtils::getDefaultDataAllocator());
    _buildFinalIndexes(final_indexes, indexes);
    m_base_copier.copyFromBufferAsync(final_indexes, buffer, var_value);
  }

  void copyToBufferAsync(ConstArrayView<Int32> indexes,
                         MutableMemoryView buffer,
                         ConstMemoryView var_value) override
  {
    UniqueArray<Int32> final_indexes(MemoryUtils::getDefaultDataAllocator());
    _buildFinalIndexes(final_indexes, indexes);
    m_base_copier.copyToBufferAsync(final_indexes, buffer, var_value);
  }
  void barrier() override { m_base_copier.barrier(); }

  void setRunQueue(const RunQueue& queue) override { m_base_copier.setRunQueue(queue); }

 private:

  GroupIndexTable* m_table = nullptr;
  DirectBufferCopier m_base_copier;

 private:

  void _buildFinalIndexes(Array<Int32>& final_indexes, ConstArrayView<Int32> orig_indexes)
  {
    // TODO: do this allocation only once and keep it.
    // Check to allocate on the device if we are on GPU (in this case, it
    // would need to run the following code on the device as well)
    GroupIndexTable& table = *m_table;
    Int32 n = orig_indexes.size();
    final_indexes.resize(n);
    for (Int32 i = 0; i < n; ++i)
      final_indexes[i] = table[orig_indexes[i]];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
