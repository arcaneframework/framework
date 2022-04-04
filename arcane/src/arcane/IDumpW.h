// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDumpW.h                                                    (C) 2000-2011 */
/*                                                                           */
/* Wrapper de IDataWriter sous l'ancienne interface IDumpW.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDUMPW_H
#define ARCANE_IDUMPW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/std/DumpW.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef NO_USER_WARNING
#warning "IDumpW is an obsolete interface which has been replaced by IDataWriter. IDumpW now uses the temporary interface wrapper DumpW."
#endif /* NO_USER_WARNING */
typedef DumpW IDumpW;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_IDUMPW_H */
