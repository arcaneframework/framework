// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExecutionStatsDumper.h                                      (C) 2000-2022 */
/*                                                                           */
/* Ecriture des statistiques d'exécution.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_EXECUTIONSTATSDUMPER_H
#define ARCANE_IMPL_EXECUTIONSTATSDUMPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture des statistiques d'exécution.
 *
 * Les statistiques sont sorties à la fois dans le listing et dans les
 * logs.
 */
class ExecutionStatsDumper
: public TraceAccessor
{
 public:

  explicit ExecutionStatsDumper(ITraceMng* trace)
  : TraceAccessor(trace)
  {}

 public:

  void dumpStats(ISubDomain* sd, ITimeStats* time_stats);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
