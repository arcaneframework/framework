// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedVariable.h                                            (C) 2000-2025 */
/*                                                                           */
/* Classe gérant une vue partagée d'une variable.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SHAREDVARIABLE_H
#define ARCANE_CORE_SHAREDVARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! Variable partagée à partir d'une variable Arcane 
 *  L'implémentation préliminaire suppose que le uniqueId des items 
 *  est le même entre sous-maillage et maillage support.
 */
template<typename ItemTypeT, typename DataTypeT>
class SharedMeshVariableScalarRefT {
public:
  typedef ItemTypeT ItemType;
  typedef DataTypeT DataType;
  typedef DataTypeT & DataTypeReturnReference;
  typedef MeshVariableScalarRefT<ItemTypeT,DataTypeT> TrueVariable;
  typedef SharedMeshVariableScalarRefT<ItemTypeT,DataTypeT> ThisVariable;
  
public:
  SharedMeshVariableScalarRefT()
    : m_true_variable(nullptr)
    , m_family(NULL)
    , m_parent_family(NULL)
    , m_direct_access(false)
    , m_family_depth(0) 
  {
    ;
  }

//   SharedMeshVariableScalarRefT(TrueVariable & v)
//     : m_true_variable(v)
//     , m_family(v.itemGroup().itemFamily())
//     , m_parent_family(NULL)
//     , m_direct_access(true)
//     , m_family_depth(0)
//   {
//     ;
//   }

  SharedMeshVariableScalarRefT(IItemFamily * family, TrueVariable & v)
    : m_true_variable(v)
    , m_family(family)
    , m_parent_family(family->parentFamily())
    , m_direct_access(false)
    , m_family_depth(-1)
 {
   IItemFamily * variable_family = v.itemGroup().itemFamily();

   // Ne gère actuellement qu'au plus un niveau d'imbrication
   // Si parent_family == family, n'utilise pas l'imbrication
   //   Attn alors si item.parent()!=item
   if (variable_family == m_family)
     m_direct_access = true;
   else if (m_parent_family == variable_family)
     m_family_depth = 0;
   else
     throw FatalErrorException(A_FUNCINFO,"Incompatible Family on shared variable");
  }

  SharedMeshVariableScalarRefT(const ThisVariable & v) 
    : m_true_variable(v.m_true_variable)
    , m_family(v.m_family)
    , m_parent_family(v.m_parent_family)
    , m_direct_access(v.m_direct_access)
    , m_family_depth(v.m_family_depth)
  {
    ;
  }

  ~SharedMeshVariableScalarRefT()
  {
    ;
  }

  DataTypeReturnReference operator[](const ItemType & i) 
  {
    ARCANE_ASSERT((m_family!=m_parent_family || i.itemBase()==i.itemBase().parentBase(m_family_depth)),("Confusion: item parent differs from item"));
    return m_true_variable.asArray()[(m_direct_access)?i.localId():i.itemBase().parentId(m_family_depth)];
  }

  DataType operator[](const ItemType & i) const
  { 
    ARCANE_ASSERT((m_family!=m_parent_family || i.itemBase()==i.itemBase().parentBase(m_family_depth)),("Confusion: item parent differs from item"));
    return m_true_variable.asArray()[(m_direct_access)?i.localId():i.itemBase().parentId(m_family_depth)];
  }

  
  DataTypeReturnReference operator[](const ItemEnumeratorT<ItemType> & i) 
  {
    ARCANE_ASSERT((m_family!=m_parent_family || (*i).itemBase()==i->parent(m_family_depth)),("Confusion: item parent differs from item"));
    return m_true_variable.asArray()[(m_direct_access)?i.localId():i->itemBase().parentId(m_family_depth)];
  }

  DataType operator[](const ItemEnumeratorT<ItemType> & i) const
  { 
    ARCANE_ASSERT((m_family!=m_parent_family || (*i).internal()==i->parent(m_family_depth)),("Confusion: item parent differs from item"));
    return m_true_variable.asArray()[(m_direct_access)?i.localId():i->internal()->parentId(m_family_depth)];
  }

  TrueVariable & trueVariable() 
  {
    return m_true_variable;
  }

  const TrueVariable & trueVariable() const
  {
    return m_true_variable;
  }
 public:
  //! TODO GG: il faudra supprimer l'opérateur d'assignement.
  ARCANE_DEPRECATED_240 void operator=(const ThisVariable & v)
  {
    m_true_variable.refersTo(v.m_true_variable);
    m_family = v.m_family;
    m_parent_family = v.m_parent_family;
    m_direct_access = v.m_direct_access;
    m_family_depth = v.m_family_depth;
  }

protected:
  TrueVariable m_true_variable;
  IItemFamily * m_family;
  IItemFamily * m_parent_family;
  bool m_direct_access;
  Integer m_family_depth;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataTypeT>
class SharedItemVariableScalarRefT {
public:
  typedef DataTypeT DataType;
  typedef DataTypeT & DataTypeReturnReference;
  typedef ItemVariableScalarRefT<DataTypeT> TrueVariable;
  typedef SharedItemVariableScalarRefT<DataTypeT> ThisVariable;
  
public:
//   SharedItemVariableScalarRefT(TrueVariable & v)
//     : m_true_variable(v)
//     , m_family(v.itemGroup().itemFamily())
//     , m_parent_family(NULL)
//     , m_direct_access(true)
//     , m_family_depth(0)
//   {
//     ;
//   }

  SharedItemVariableScalarRefT(IItemFamily * family, TrueVariable & v)
    : m_true_variable(v)
    , m_family(family)
    , m_parent_family(family->parentFamily())
    , m_direct_access(false)
    , m_family_depth(-1)
 {
   IItemFamily * variable_family = v.itemGroup().itemFamily();

   // Ne gère actuellement qu'au plus un niveau d'imbrication
   // Si parent_family == family, n'utilise pas l'imbrication
   //   Attn alors si item.parent()!=item
   if (variable_family == m_family)
     m_direct_access = true;
   else if (m_parent_family == variable_family)
     m_family_depth = 0;
   else
     throw FatalErrorException(A_FUNCINFO,"Incompatible Family on shared variable");
  }

  SharedItemVariableScalarRefT(const ThisVariable & v) 
    : m_true_variable(v.m_true_variable)
    , m_family(v.m_family)
    , m_parent_family(v.m_parent_family)
    , m_direct_access(v.m_direct_access)
    , m_family_depth(v.m_family_depth)
  {
    ;
  }

  ~SharedItemVariableScalarRefT()
  {
    ;
  }

  DataTypeReturnReference operator[](const Item & i) 
  {
    ARCANE_ASSERT((m_family!=m_parent_family || i.itemBase()==i.itemBase().parentBase(m_family_depth)),("Confusion: item parent differs from item"));
    return m_true_variable.asArray()[(m_direct_access)?i.localId():i.itemBase().parentId(m_family_depth)];
  }

  DataType operator[](const Item & i) const
  { 
    ARCANE_ASSERT((m_family!=m_parent_family || i.itemBase()==i.itemBase().parentBase(m_family_depth)),("Confusion: item parent differs from item"));
    return m_true_variable.asArray()[(m_direct_access)?i.localId():i.itemBase().parentId(m_family_depth)];
  }

  
  DataTypeReturnReference operator[](const ItemEnumerator & i) 
  {
    ARCANE_ASSERT((m_family!=m_parent_family || (*i).itemBase()==i->parent(m_family_depth)),("Confusion: item parent differs from item"));
    return m_true_variable.asArray()[(m_direct_access)?i.localId():i->itemBase().parentId(m_family_depth)];
  }

  DataType operator[](const ItemEnumerator & i) const
  { 
    ARCANE_ASSERT((m_family!=m_parent_family || (*i).itemBase()==i->parent(m_family_depth)),("Confusion: item parent differs from item"));
    return m_true_variable.asArray()[(m_direct_access)?i.localId():i->itemBase().parentId(m_family_depth)];
  }

  TrueVariable & trueVariable() 
  {
    return m_true_variable;
  }

  const TrueVariable & trueVariable() const
  {
    return m_true_variable;
  }
 public:
  //! TODO GG: il faudra supprimer l'opérateur d'assignement.
  ARCANE_DEPRECATED_240 void operator=(const ThisVariable & v)
  {
    m_true_variable.refersTo(v.m_true_variable);
    m_family = v.m_family;
    m_parent_family = v.m_parent_family;
    m_direct_access = v.m_direct_access;
    m_family_depth = v.m_family_depth;
  }
protected:
  TrueVariable m_true_variable;
  IItemFamily * m_family;
  IItemFamily * m_parent_family;
  bool m_direct_access;
  Integer m_family_depth;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_SHARED_VARIABLE_H */

