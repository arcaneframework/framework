// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeMetricCollector.h                                      (C) 2000-2025 */
/*                                                                           */
/* Interface gérant les statistiques sur les temps d'exécution.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_ITIMEMETRICCOLLECTOR_H
#define ARCCORE_TRACE_ITIMEMETRICCOLLECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceGlobal.h"
#include "arccore/base/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * API en cours de définition. Ne pas utiliser en dehors de Arccore/Arcane.
 */

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface gérant les statistiques sur l'exécution.
 */
class ARCCORE_TRACE_EXPORT ITimeMetricCollector
{
 public:

 public:

  // Libère les ressources.
  virtual ~ITimeMetricCollector() = default;

 public:

  virtual TimeMetricAction getAction(const TimeMetricActionBuildInfo& x) =0;
  virtual TimeMetricId beginAction(const TimeMetricAction& handle) =0;
  virtual void endAction(const TimeMetricId& metric_id) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
