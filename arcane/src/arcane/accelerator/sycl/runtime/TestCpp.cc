// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestCpp.cc                                                  (C) 2000-2022 */
/*                                                                           */
/* Fichier de tests pour HIP.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

extern "C"
int arcaneTestSycl1();
extern "C"
int arcaneTestSycl2();

extern "C" ARCANE_EXPORT
int func0()
{
  arcaneTestSycl1();
  arcaneTestSycl2();
  return 0;
}
