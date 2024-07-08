// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdScalarMeshVariables.h                                    (C) 2000-2024 */
/*                                                                           */
/* Définition de variables scalaires du maillage pour des tests.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TEST_STDSCALARMESHVARIABLES_H
#define ARCANE_TEST_STDSCALARMESHVARIABLES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/StdMeshVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Définition de variables scalaires du maillage pour des tests.
*/
template<typename ItemType>
class StdScalarMeshVariables
: public StdMeshVariables< StdMeshVariableTraits2<ItemType,0> >
{
 public:

   typedef typename ItemTraitsT<ItemType>::ItemGroupType GroupType;

 public:

  StdScalarMeshVariables(const MeshHandle& mesh_handle,const String& basestr);
  StdScalarMeshVariables(const MeshHandle& mesh_handle,const String& basestr,const String& family_name);

 public:

  void initialize(){}
  void setValues(Integer iteration, const GroupType& group);
  void setValuesWithViews(Integer iteration, const GroupType& group);
  Integer checkValues(Integer iteration, const GroupType& group);
  Integer checkValuesWithViews(Integer iteration, const GroupType& group);
  Integer checkReplica();

  void setEvenValues(Integer iteration, const GroupType& group);
  void setOddValues(Integer iteration, const GroupType& group);
  Integer checkGhostValuesOddOrEven(Integer iteration, const GroupType& group);
  
 private:

  void setItemValues(Int64 n, ItemType item);
  
  Integer m_nb_display_error;
  Integer m_nb_displayed_error;

  Integer _checkItemValues(Integer iteration, ItemType item,
                           const MultiScalarValue& item_value);
  Integer _checkValue(ItemType item,const MultiScalarValue& ref_value,
                      const MultiScalarValue& item_value);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

