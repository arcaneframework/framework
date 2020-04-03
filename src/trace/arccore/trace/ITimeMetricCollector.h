// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* ITimeMetricCollector.h                                      (C) 2000-2020 */
/*                                                                           */
/* Interface gérant les statistiques sur les temps d'exécution.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_IMETRICCOLLECTOR_H
#define ARCCORE_TRACE_IMETRICCOLLECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceGlobal.h"
#include "arccore/base/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * API en cours de définition. Ne pas utiliser en dehors de Arccore/Arcane.
 */

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface gérant les statistiques sur l'exécution.
 */
class ARCCORE_BASE_EXPORT ITimeMetricCollector
{
 public:

 public:

  // Libère les ressources.
  virtual ~ITimeMetricCollector() = default;

 public:

  virtual TimeMetricActionHandle getHandle(const TimeMetricActionHandleBuildInfo& x) =0;
  virtual TimeMetricId beginAction(const TimeMetricActionHandle& handle) =0;
  virtual void endAction(const TimeMetricId& metric_id) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
