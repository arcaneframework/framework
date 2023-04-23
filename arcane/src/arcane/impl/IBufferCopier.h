// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IBufferCopier.h                                             (C) 2000-2023 */
/*                                                                           */
/* Interface pour la copie de buffer.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_IBUFFERCOPIER_H
#define ARCANE_IMPL_IBUFFERCOPIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryView.h"

#include "arcane/core/GroupIndexTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface pour copier des éléments entre deux zones avec indexation.
 */
class IBufferCopier
{
 public:

  virtual ~IBufferCopier() = default;

 public:

  virtual void copyFromBuffer(Int32ConstArrayView indexes,
                              ConstMemoryView buffer,
                              MutableMemoryView var_value) = 0;

  virtual void copyToBuffer(Int32ConstArrayView indexes,
                            MutableMemoryView buffer,
                            ConstMemoryView var_value) = 0;

  virtual IMemoryAllocator* allocator() const = 0;

 public:

  virtual void setRunQueue(RunQueue* queue) = 0;
  virtual void setAllocator(IMemoryAllocator* allocator) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DirectBufferCopier
: public IBufferCopier
{
 public:

  void copyFromBuffer(Int32ConstArrayView indexes,
                      ConstMemoryView buffer,
                      MutableMemoryView var_value) override
  {
    buffer.copyToIndexes(var_value, indexes, m_queue);
  }

  void copyToBuffer(Int32ConstArrayView indexes,
                    MutableMemoryView buffer,
                    ConstMemoryView var_value) override
  {
    buffer.copyFromIndexes(var_value, indexes, m_queue);
  }

  IMemoryAllocator* allocator() const override { return m_allocator; }
  void setRunQueue(RunQueue* queue) override { m_queue = queue; }
  void setAllocator(IMemoryAllocator* allocator) override { m_allocator = allocator; }

 private:

  RunQueue* m_queue = nullptr;
  IMemoryAllocator* m_allocator = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TableBufferCopier
: public IBufferCopier
{
 public:

  TableBufferCopier(GroupIndexTable* table)
  : m_table(table)
  {}

  void copyFromBuffer(Int32ConstArrayView indexes,
                      ConstMemoryView buffer,
                      MutableMemoryView var_value) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes, indexes);
    m_base_copier.copyFromBuffer(final_indexes, buffer, var_value);
  }

  void copyToBuffer(Int32ConstArrayView indexes,
                    MutableMemoryView buffer,
                    ConstMemoryView var_value) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes, indexes);
    m_base_copier.copyToBuffer(final_indexes, buffer, var_value);
  }
  IMemoryAllocator* allocator() const override { return m_base_copier.allocator(); }
  void setRunQueue(RunQueue* queue) override { m_base_copier.setRunQueue(queue); }
  void setAllocator(IMemoryAllocator* allocator) override { m_base_copier.setAllocator(allocator); }

 private:

  GroupIndexTable* m_table;
  DirectBufferCopier m_base_copier;

 private:

  void _buildFinalIndexes(Array<Int32>& final_indexes, ConstArrayView<Int32> orig_indexes)
  {
    // TODO: faire cette allocation qu'une seule fois et la conserver.
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
