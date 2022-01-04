// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiBatchMainFactory.cc                                      (C) 2000-2005 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant MPI.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"

#include "arcane/parallel/IStat.h"

#include "arcane/parallel/mpi/MpiBatchMainFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IArcaneMain*
createArcaneMainBatch(const ApplicationInfo& app_info,IMainFactory*);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IArcaneMain* MpiBatchMainFactory::
createArcaneMain(const ApplicationInfo& app_info)
{ 
	return createArcaneMainBatch(app_info,this);
}

int MpiBatchMainFactory::
exec(const ApplicationInfo& app_info)
{
	MpiBatchMainFactory pbmf;
	int r = Arcane::ArcaneMain::arcaneMain(app_info,&pbmf);
	return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
