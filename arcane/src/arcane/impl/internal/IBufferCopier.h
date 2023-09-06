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
 *
 * Les méthodes de copie peuvent être asynchrones. Il faut appeler barrier()
 * pour s'assurer que ces copies sont bien terminées avant d'utilier les
 * valeurs des buffers.
 */
class IBufferCopier
{
 public:

  virtual ~IBufferCopier() = default;

 public:

  virtual void copyFromBufferAsync(Int32ConstArrayView indexes,
                                   ConstMemoryView buffer,
                                   MutableMemoryView var_value) = 0;

  virtual void copyToBufferAsync(Int32ConstArrayView indexes,
                                 MutableMemoryView buffer,
                                 ConstMemoryView var_value) = 0;

  //! Bloque tant que les copies ne sont pas terminées.
  virtual void barrier() = 0;

 public:

  virtual void setRunQueue(RunQueue* queue) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DirectBufferCopier
: public IBufferCopier
{
 public:

  void copyFromBufferAsync(Int32ConstArrayView indexes,
                           ConstMemoryView buffer,
                           MutableMemoryView var_value) override
  {
    buffer.copyToIndexes(var_value, indexes, m_queue);
  }

  void copyToBufferAsync(Int32ConstArrayView indexes,
                         MutableMemoryView buffer,
                         ConstMemoryView var_value) override
  {
    buffer.copyFromIndexes(var_value, indexes, m_queue);
  }

  void barrier() override;
  void setRunQueue(RunQueue* queue) override { m_queue = queue; }

 private:

  RunQueue* m_queue = nullptr;
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

  void copyFromBufferAsync(Int32ConstArrayView indexes,
                           ConstMemoryView buffer,
                           MutableMemoryView var_value) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes, indexes);
    m_base_copier.copyFromBufferAsync(final_indexes, buffer, var_value);
  }

  void copyToBufferAsync(Int32ConstArrayView indexes,
                         MutableMemoryView buffer,
                         ConstMemoryView var_value) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes, indexes);
    m_base_copier.copyToBufferAsync(final_indexes, buffer, var_value);
  }
  void barrier() override { m_base_copier.barrier(); }

  void setRunQueue(RunQueue* queue) override { m_base_copier.setRunQueue(queue); }

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
