// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckpointService.cc                                        (C) 2000-2007 */
/*                                                                           */
/* Service de protection/reprise.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/CheckpointService.h"

#include "arcane/IDataReader2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CheckpointService::
CheckpointService(const ServiceBuildInfo& sbi)
: BasicService(sbi)
, m_current_time(-1.)
, m_current_index(-1)
{
}

void CheckpointService::
setCheckpointTimes(RealConstArrayView times)
{
  m_checkpoint_times = RealUniqueArray(times);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointService::
setCurrentTimeAndIndex(Real current_time,Integer current_index)
{
  m_current_time = current_time;
  m_current_index = current_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
