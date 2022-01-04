// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IBufferCopier.h                                             (C) 2000-2021 */
/*                                                                           */
/* Interface pour la copie de buffer.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_IBUFFERCOPIER_H
#define ARCANE_IMPL_IBUFFERCOPIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/GroupIndexTable.h"

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

template<typename SimpleType>
class DirectBufferCopier
: public IBufferCopier<SimpleType>
{
 public:
  
  void copyFromBufferOne(Int32ConstArrayView indexes,
                         ConstArrayView<SimpleType> buffer,
                         ArrayView<SimpleType> var_value) override
  {
    Integer n = indexes.size();
#ifdef ARCANE_CHECK
    for( Integer i=0; i<n; ++i )
      var_value[indexes[i]] = buffer[i];
#else
    const SimpleType* ARCANE_RESTRICT r_buf = buffer.data();
    const Int32* ARCANE_RESTRICT r_idx = indexes.data();
    SimpleType* ARCANE_RESTRICT r_var_value = var_value.data();
    for( Integer i=0; i<n; ++i )
      r_var_value[r_idx[i]] = r_buf[i];
#endif 
  }
  
  void copyToBufferOne(Int32ConstArrayView indexes,
                       ArrayView<SimpleType> buffer,
                       ConstArrayView<SimpleType> var_value) override
  { 
    Integer n = indexes.size();
#ifdef ARCANE_CHECK
    for( Integer i=0; i<n; ++i )
      buffer[i] = var_value[indexes[i]];
#else
    SimpleType* ARCANE_RESTRICT r_buf = buffer.data();
    const Int32* ARCANE_RESTRICT r_idx = indexes.data();
    const SimpleType* ARCANE_RESTRICT r_var_value = var_value.data();
    for( Integer i=0; i<n; ++i )
      r_buf[i] = r_var_value[r_idx[i]];
#endif
  }

  void copyFromBufferMultiple(Int32ConstArrayView indexes,
                              ConstArrayView<SimpleType> buffer,
                              ArrayView<SimpleType> var_value,
                              Integer dim2_size) override
  {
    Integer n = indexes.size();
    for( Integer i=0; i<n; ++i ){
      Integer zindex = i*dim2_size;
      Integer zci = indexes[i]*dim2_size;
      for( Integer z=0; z<dim2_size; ++ z )
        var_value[zci+z] = buffer[zindex+z];
    }
  }

  void copyToBufferMultiple(Int32ConstArrayView indexes,
                            ArrayView<SimpleType> buffer,
                            ConstArrayView<SimpleType> var_value,
                            Integer dim2_size) override
  {
    Integer n = indexes.size();
    for( Integer i=0; i<n; ++i ){
      Integer zindex = i*dim2_size;
      Integer zci = indexes[i]*dim2_size;
      for( Integer z=0; z<dim2_size; ++ z )
        buffer[zindex+z] = var_value[zci+z];
    }
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
  ~TableBufferCopier() {}

  void copyFromBufferOne(Int32ConstArrayView indexes,
                         ConstArrayView<SimpleType> buffer,
                         ArrayView<SimpleType> var_value) override
  {
    Integer n = indexes.size();
    GroupIndexTable& table = *m_table;
    for( Integer i=0; i<n; ++i ) 
      var_value[table[indexes[i]]] = buffer[i];
  }
  
  void copyToBufferOne(Int32ConstArrayView indexes,
                       ArrayView<SimpleType> buffer,
                       ConstArrayView<SimpleType> var_value) override
  { 
    Integer n = indexes.size();
    GroupIndexTable& table = *m_table;
    for( Integer i=0; i<n; ++i )
      buffer[i] = var_value[table[indexes[i]]];
  }
   
  void copyFromBufferMultiple(Int32ConstArrayView indexes,
                              ConstArrayView<SimpleType> buffer,
                              ArrayView<SimpleType> var_value,
                              Integer dim2_size) override
  {
    Integer n = indexes.size();
    GroupIndexTable& table = *m_table;
    for( Integer i=0; i<n; ++i ){
      Integer zindex = i*dim2_size;
      Integer zci = table[indexes[i]]*dim2_size;
      for( Integer z=0; z<dim2_size; ++ z )
        var_value[zci+z] = buffer[zindex+z];
    }
  }

  void copyToBufferMultiple(Int32ConstArrayView indexes,
                            ArrayView<SimpleType> buffer,
                            ConstArrayView<SimpleType> var_value,
                            Integer dim2_size) override
  {
    Integer n = indexes.size();
    GroupIndexTable& table = *m_table;
    for( Integer i=0; i<n; ++i ){
      Integer zindex = i*dim2_size;
      Integer zci = table[indexes[i]]*dim2_size;
      for( Integer z=0; z<dim2_size; ++ z )
        buffer[zindex+z] = var_value[zci+z];
    }
  }
 private:
  GroupIndexTable* m_table;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
