// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDumpRW.h                                                   (C) 2000-2011 */
/*                                                                           */
/* Wrapper de IDataReaderWriter sous l'ancienne interface IDumpRW.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDUMPRW_H
#define ARCANE_IDUMPRW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/std/DumpRW.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef NO_USER_WARNING
#warning "IDumpRW is an obsolete interface which has been replaced by IDataReaderWriter. IDumpRW now uses the temporary interface wrapper DumpRW."
#endif /* NO_USER_WARNING */
typedef DumpRW IDumpRW;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_IDUMPRW_H */
