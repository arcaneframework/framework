// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemVariable.h                                           (C) 2000-2025 */
/*                                                                           */
/* Aggregated variable of arbitrary types.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEMVARIABLE_H 
#define ARCANE_CORE_ANYITEM_ANYITEMVARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

#include "arcane/core/anyitem/AnyItemGlobal.h"
#include "arcane/core/anyitem/AnyItemGroup.h"
#include "arcane/core/anyitem/AnyItemLinkFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AnyItem
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Aggregated variable of arbitrary types (no array variables)
 *
 * For example:
 *
 * AnyItem::Family family;
 *
 * family << AnyItem::GroupBuilder( allFaces() ) 
 *        << AnyItem::GroupBuilder( allCells() );
 *
 * AnyItem::Variable<Real> variable(family);
 *
 * variable[allFaces()] << m_face_variable;
 * variable[allCells()] << m_cell_variable;
 *
 * Real value = 0.;
 * ENUMERATE_ANY_ITEM(iitem, family.allItems()) {
 *   value += variable[iitem];
 * }
 *
 */
template<typename DataType>
class Variable : public IFamilyObserver
{
public:
  
  /*!
   * \brief Tool for adding a variable to a group
   */
  class VariableAdder
  {
  public:
    
    VariableAdder(Variable<DataType>& variable, const ItemGroup& group) 
      : m_variable(variable)
      , m_group(group)
      , m_used(false) {}
    
    ~VariableAdder() 
    {
      ARCANE_ASSERT((m_used == true),("VariableAdder never used"));
    }
    
    //! Binding of a variable 
    template<typename K, typename T>  
    inline void operator<<(MeshVariableScalarRefT<K,T>& v) 
    {
      ARCANE_ASSERT((m_used == false),("VariableAdder already used"));
      m_variable._insertVariable(m_group,v.asArray());
      m_variable._insertInternalVariable(m_group,v.variable());
      m_used = true;
    }
    
    //! Binding of a partial variable
    template<typename K, typename T> 
    inline void operator<<(MeshPartialVariableScalarRefT<K,T>& v) 
    {
      ARCANE_ASSERT((m_used == false),("VariableAdder already used"));
      m_variable._insertPartialVariable(m_group,v.asArray());
      m_variable._insertInternalVariable(m_group,v.variable());
      m_used = true;
    }
  private:
    
    //! AnyItem Variable
    Variable<DataType>& m_variable;
    
    //! Variable group
    const ItemGroup& m_group;
    
    //! Indicator of Adder usage 
    bool m_used;
  };
  
public:
  
  Variable(const Family& family) 
    : m_family(family)
    , m_values(m_family.groupSize())
    , m_variables(m_family.groupSize()) 
  {
    // The family registers the carried variables
    m_family.registerObserver(*this);
    for(Integer i = 0; i < m_family.groupSize(); ++i)
      m_variables[i] = NULL;
  }
  
  Variable(const Variable& v) 
    : m_family(v.m_family)
    , m_values(v.m_values)
    , m_variables(v.m_variables) 
  {
    // The family registers the carried variables
    m_family.registerObserver(*this);
  }
  
  ~Variable() 
  {
    // The family deregisters the variable
    arcaneCallFunctionAndTerminateIfThrow([&]() { m_family.removeObserver(*this); });
  }
  
  //! Direct accessor by an AnyItem enumerator
  inline DataType& operator[](const Group::BlockItemEnumerator& item) { 
    return m_values[item.groupIndex()][item.varIndex()];
  }
  
  //! Direct accessor by an AnyItem enumerator
  inline const DataType& operator[](const Group::BlockItemEnumerator & item) const { 
    return m_values[item.groupIndex()][item.varIndex()];
  }
  
  //! Direct accessor by a LinkFamily element (LinkData)
  inline DataType& operator[](const LinkFamily::LinkData & item) { 
    return m_values[item.groupIndex()][item.varIndex()];
  }
  
  //! Direct accessor by a LinkFamily element (LinkData)
  inline const DataType& operator[](const LinkFamily::LinkData & item) const { 
    return m_values[item.groupIndex()][item.varIndex()];
  }

  //! Adding a variable for a group
  inline VariableAdder operator[](const ItemGroup& group) 
  {
    return VariableAdder(*this,group);
  }

  template<typename T>
  inline VariableAdder operator[](const ItemGroupT<T>& group) 
  {
    return VariableAdder(*this,group);
  }

  //! Accessor to the family
  inline const Family& family() const { return m_family; }

  //! Array of variables
  inline ConstArrayView< IVariable* > variables() const 
  {
    return m_variables;
  } 
  
  //! Raw data associated with a group identified relative to its family
  inline ArrayView<DataType> valuesAtGroup(const Integer igrp) 
  {
    return m_values[igrp];
  }

  //! Raw data associated with a group identified relative to its family
  inline ConstArrayView<DataType> valuesAtGroup(const Integer igrp) const
  {
    return m_values[igrp];
  }

  //! Notification of family invalidation
  inline void notifyFamilyIsInvalidate() {
    // If the family changes, we invalidate the variables and resize
    m_values.resize(m_family.groupSize());
    m_variables.resize(m_family.groupSize());
    for(Integer i = 0; i < m_family.groupSize(); ++i)
      m_variables[i] = NULL; 
  }

  //! Notification of family enlargement
  inline void notifyFamilyIsIncreased() {
    // If the family is enlarged, we simply resize
    const Integer old_size = m_values.size();
    ARCANE_ASSERT((old_size < m_family.groupSize()),("Old size greater than new size!"));
    m_values.resize(m_family.groupSize());
    m_variables.resize(m_family.groupSize());
    for(Integer i = old_size; i < m_family.groupSize(); ++i)
      m_variables[i] = NULL; 
  }
  
private:
  
  inline void _insertVariable(ItemGroup group, ArrayView<DataType> v) 
  {
    if(m_family.isPartial(group))
      throw FatalErrorException(String::format("Group '{0}' defined partial",group.name())); 
    m_values[m_family.groupIndex(group)] = v;
  }

  inline void _insertPartialVariable(ItemGroup group, ArrayView<DataType> v) 
  {
    if(not m_family.isPartial(group))
      throw FatalErrorException(String::format("Group '{0}' not defined partial",group.name())); 
    m_values[m_family.groupIndex(group)] = v;
  }
  
  inline void _insertInternalVariable(ItemGroup group, IVariable* v) 
  {
    m_variables[m_family.groupIndex(group)] = v;
  }

private:
  
  //! AnyItem Family of groups
  const Family m_family;

  //! Container of generic variables
  Arcane::UniqueArray< ArrayView<DataType> > m_values;

  //! Container of variables
  Arcane::UniqueArray< IVariable* > m_variables;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::AnyItem

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ANYITEMS_ANYITEMVARIABLE_H */
