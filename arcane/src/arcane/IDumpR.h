// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDumpR.h                                                    (C) 2000-2011 */
/*                                                                           */
/* Wrapper de IDataReader sous l'ancienne interface IDumpR.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDUMPR_H
#define ARCANE_IDUMPR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/std/DumpR.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef NO_USER_WARNING
#warning "IDumpR is an obsolete interface which has been replaced by IDataReader. IDumpR now uses the temporary interface wrapper DumpR."
#endif /* NO_USER_WARNING */
typedef DumpR IDumpR;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_IDUMPR_H */
