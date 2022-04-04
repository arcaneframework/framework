// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Collection.cc                                               (C) 2000-2008 */
/*                                                                           */
/* Classe de base d'une collection.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Collection.h"
#include "arcane/utils/CollectionImpl.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/TraceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCANE_UTILS_EXPORT void
throwOutOfRangeException()
{
  cerr << "** FATAL: Invalid access on a collection (array, list, ...).\n";
  throw ArgumentException(A_FUNCINFO,"Bad index");
  //arcaneDebugPause("throwOutOfRangeException");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

