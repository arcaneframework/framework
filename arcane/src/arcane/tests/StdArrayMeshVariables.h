// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdArrayMeshVariables.h                                     (C) 2000-2024 */
/*                                                                           */
/* Définition de variables tableaux du maillage pour des tests.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TEST_STDARRAYMESHVARIABLES_H
#define ARCANE_TEST_STDARRAYMESHVARIABLES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/StdMeshVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Définition de variables tableaux du maillage pour des tests.
*/
template<typename ItemType>
class StdArrayMeshVariables
: public StdMeshVariables< StdMeshVariableTraits2<ItemType,1> >
{
 public:

  typedef typename ItemTraitsT<ItemType>::ItemGroupType GroupType;

 public:
  
  StdArrayMeshVariables(const MeshHandle& mesh_handle,const String& basestr);
  StdArrayMeshVariables(const MeshHandle& mesh_handle,const String& basestr,const String& family_name);

 public:

  void initialize();
  Integer nbValuePerItem();
  void setValue(Int64 n, ItemType item);
  void setValues(Integer iteration, const GroupType& group);
  void setEvenValues(Integer iteration, const GroupType& group);
  void setOddValues(Integer iteration, const GroupType& group);
  Integer checkValues(Integer iteration, const GroupType& group);
  Integer checkGhostValuesOddOrEven(Integer iteration, const GroupType& group);
  Integer checkReplica();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

