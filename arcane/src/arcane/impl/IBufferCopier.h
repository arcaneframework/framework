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
template<typename SimpleType>
class IBufferCopier 
{
 public:
  
  virtual ~IBufferCopier() = default;

  virtual void copyFromBufferOne(Int32ConstArrayView indexes,
                                 ConstArrayView<SimpleType> buffer,
                                 ArrayView<SimpleType> var_value) = 0;

  virtual void copyToBufferOne(Int32ConstArrayView indexes,
                               ArrayView<SimpleType> buffer,
                               ConstArrayView<SimpleType> var_value) = 0;  

  virtual void copyFromBufferMultiple(Int32ConstArrayView indexes,
                                      ConstArrayView<SimpleType> buffer,
                                      ArrayView<SimpleType> var_value,
                                      Integer dim2_size) = 0;
  
  virtual void copyToBufferMultiple(Int32ConstArrayView indexes,
                                    ArrayView<SimpleType> buffer,
                                    ConstArrayView<SimpleType> var_value,
                                    Integer dim2_size) = 0;
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

  void copyFromBufferOne(Int32ConstArrayView indexes,
                         ConstArrayView<SimpleType> buffer,
                         ArrayView<SimpleType> var_value) override
  {
    MemoryView from(buffer);
    MutableMemoryView to(var_value);
    from.copyToIndexesHost(to,indexes);
  }

  void copyToBufferOne(Int32ConstArrayView indexes,
                       ArrayView<SimpleType> buffer,
                       ConstArrayView<SimpleType> var_value) override
  {
    MemoryView from(var_value);
    MutableMemoryView to(buffer);
    to.copyFromIndexesHost(from,indexes);
  }

  void copyFromBufferMultiple(Int32ConstArrayView indexes,
                              ConstArrayView<SimpleType> buffer,
                              ArrayView<SimpleType> var_value,
                              Integer dim2_size) override
  {
    MemoryView from(buffer,dim2_size);
    MutableMemoryView to(var_value,dim2_size);
    from.copyToIndexesHost(to,indexes);
  }

  void copyToBufferMultiple(Int32ConstArrayView indexes,
                            ArrayView<SimpleType> buffer,
                            ConstArrayView<SimpleType> var_value,
                            Integer dim2_size) override
  {
    MemoryView from(var_value,dim2_size);
    MutableMemoryView to(buffer,dim2_size);
    to.copyFromIndexesHost(from,indexes);
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

  void copyFromBufferOne(Int32ConstArrayView indexes,
                         ConstArrayView<SimpleType> buffer,
                         ArrayView<SimpleType> var_value) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes,indexes);
    m_base_copier.copyFromBufferOne(final_indexes,buffer,var_value);
  }

  void copyToBufferOne(Int32ConstArrayView indexes,
                       ArrayView<SimpleType> buffer,
                       ConstArrayView<SimpleType> var_value) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes,indexes);
    m_base_copier.copyToBufferOne(final_indexes,buffer,var_value);
  }

  void copyFromBufferMultiple(Int32ConstArrayView indexes,
                              ConstArrayView<SimpleType> buffer,
                              ArrayView<SimpleType> var_value,
                              Integer dim2_size) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes,indexes);
    m_base_copier.copyFromBufferMultiple(final_indexes,buffer,var_value,dim2_size);
  }

  void copyToBufferMultiple(Int32ConstArrayView indexes,
                            ArrayView<SimpleType> buffer,
                            ConstArrayView<SimpleType> var_value,
                            Integer dim2_size) override
  {
    UniqueArray<Int32> final_indexes;
    _buildFinalIndexes(final_indexes,indexes);
    m_base_copier.copyToBufferMultiple(final_indexes,buffer,var_value,dim2_size);
  }

 private:

  GroupIndexTable* m_table;
  DirectBufferCopier<SimpleType> m_base_copier;

 private:

  void _buildFinalIndexes(Array<Int32>& final_indexes,ConstArrayView<Int32> orig_indexes)
  {
    // TODO: utiliser des buffer de taille fixe pour ne pas avoir à faire
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
