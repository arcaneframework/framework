// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdArrayMeshVariables.h                                     (C) 2000-2024 */
/*                                                                           */
/* Definition of mesh array variables for tests.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TEST_STDARRAYMESHVARIABLES_H
#define ARCANE_TEST_STDARRAYMESHVARIABLES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/StdMeshVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \brief Definition of mesh array variables for tests.
*/
template <typename ItemType>
class StdArrayMeshVariables
: public StdMeshVariables<StdMeshVariableTraits2<ItemType, 1>>
{
 public:

  typedef typename ItemTraitsT<ItemType>::ItemGroupType GroupType;

 public:

  StdArrayMeshVariables(const MeshHandle& mesh_handle, const String& basestr);
  StdArrayMeshVariables(const MeshHandle& mesh_handle, const String& basestr, const String& family_name);

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

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
