// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneThread.h                                              (C) 2000-2020 */
/*                                                                           */
/* Fichier d'en-tête pour les threads.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_ARCANETHREAD_H
#define ARCANE_PARALLEL_THREAD_ARCANETHREAD_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
namespace MP = Arccore::MessagePassing;
using Request = MP::Request;
using IRequestCreator = MP::IRequestCreator;
using eReduceType = MP::eReduceType;
using eBlockingType = MP::eBlockingType;
using PointToPointMessageInfo = MP::PointToPointMessageInfo;
using MessageRank = MP::MessageRank;
using MessageTag = MP::MessageTag;
using MessageId = MP::MessageId;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

