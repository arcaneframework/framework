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
 */
template <typename SimpleType>
class IBufferCopier
{
 public:

  virtual ~IBufferCopier() = default;

 public:

  virtual void copyFromBuffer(Int32ConstArrayView indexes,
                              MemoryView buffer,
                              MutableMemoryView var_value) = 0;

  virtual void copyToBuffer(Int32ConstArrayView indexes,
                            MutableMemoryView buffer,
                            MemoryView var_value) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType>
class DirectBufferCopier
: public IBufferCopier<SimpleType>
{
 public:

  void copyFromBuffer(Int32ConstArrayView indexes,
                      MemoryView buffer,
                      MutableMemoryView var_value) override
  {
    buffer.copyToIndexesHost(var_value, indexes);
  }

  void copyToBuffer(Int32ConstArrayView indexes,
                    MutableMemoryView buffer,
                    MemoryView var_value) override
  {
    buffer.copyFromIndexesHost(var_value, indexes);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType>
class TableBufferCopier
: public IBufferCopier<SimpleType>
{
 public:

  TableBufferCopier(GroupIndexTable* table) : m_table(table) {}

  void copyFromBuffer(Int32ConstArrayView indexes,
                      MemoryView buffer,
                      MutableMemoryView var_value) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes, indexes);
    m_base_copier.copyFromBuffer(final_indexes, buffer, var_value);
  }

  void copyToBuffer(Int32ConstArrayView indexes,
                    MutableMemoryView buffer,
                    MemoryView var_value) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes, indexes);
    m_base_copier.copyToBuffer(final_indexes, buffer, var_value);
  }

 private:

  GroupIndexTable* m_table;
  DirectBufferCopier<SimpleType> m_base_copier;

 private:

  void _buildFinalIndexes(Array<Int32>& final_indexes,ConstArrayView<Int32> orig_indexes)
  {
    // TODO: utiliser des buffers de taille fixe pour ne pas avoir à faire
    // d'allocation
    GroupIndexTable& table = *m_table;
    Int32 n = orig_indexes.size();
    final_indexes.resize(n);
    for( Int32 i=0; i<n; ++i )
      final_indexes[i] = table[orig_indexes[i]];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
