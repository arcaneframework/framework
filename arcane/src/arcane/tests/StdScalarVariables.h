// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdMeshVariables.h                                          (C) 2000-2016 */
/*                                                                           */
/* Définition de variables scalaires pour des tests.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TEST_STDSCALARVARIABLES_H
#define ARCANE_TEST_STDSCALARVARIABLES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/VariableTypes.h"
#include "arcane/VariableBuildInfo.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StdScalarVariables
{
 public:

  StdScalarVariables(const MeshHandle& mesh_handle,const String& basestr);

 public:

  void setValues(Integer iteration);
  Integer checkValues(Integer iteration);
  Integer checkReplica();

 public:

  VariableScalarByte m_byte;
  VariableScalarReal m_real;
  VariableScalarInt64 m_int64;
  VariableScalarInt32 m_int32;
  VariableScalarInt16 m_int16;
  VariableScalarReal2 m_real2;
  VariableScalarReal2x2 m_real2x2;
  VariableScalarReal3 m_real3;
  VariableScalarReal3x3 m_real3x3;

 protected:

  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

