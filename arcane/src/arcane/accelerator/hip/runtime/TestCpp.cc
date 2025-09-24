// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestCpp.cc                                                  (C) 2000-2025 */
/*                                                                           */
/* Fichier de tests pour HIP.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

extern "C"
int arcaneTestHip1();
extern "C"
int arcaneTestHip2();
extern "C"
int arcaneTestHip3();
extern "C"
int arcaneTestHipNumArray();
extern "C"
int arcaneTestHipReduction();
extern "C"
int arcaneTestVirtualFunction();

extern "C" ARCANE_EXPORT
int func0()
{
  arcaneTestHip1();
  arcaneTestHip2();
  arcaneTestHip3();
  arcaneTestHipNumArray();
  arcaneTestHipReduction();
  arcaneTestVirtualFunction();
  return 0;
}
