// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TypesParallelTester.h                                       (C) 2000-2007 */
/*                                                                           */
/* Types du module de test du parallélisme.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TESTS_TYPESPARALLELTESTER_H
#define ARCANE_TESTS_TYPESPARALLELTESTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct TypesParallelTester
{
  enum eTestParallel
  {
    TestAll,
    TestParallelMng,
    TestLoadBalance,
    TestGetVariableValues,
    TestGhostItemsReduceOperation,
    TestTransferValues
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

