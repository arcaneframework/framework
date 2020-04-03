// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* IMetricCollector.h                                          (C) 2000-2020 */
/*                                                                           */
/* Interface gérant les statistiques sur les temps d'exécution.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_IMETRICCOLLECTOR_H
#define ARCCORE_TRACE_IMETRICCOLLECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

class IMetricCollector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_BASE_EXPORT MetricActionHandle
{
 public:
  IMetricCollector* m_collector;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_BASE_EXPORT MetricId
{
 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_BASE_EXPORT MetricSentry
{
 public:
  inline MetricSentry(const MetricActionHandle& handle);
  inline ~MetricSentry() noexcept(false);
 private:
  IMetricCollector* m_collector;
  MetricId m_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface gérant les statistiques sur l'exécution.
 */
class ARCCORE_BASE_EXPORT IMetricCollector
{
 public:

 public:

  // Libère les ressources.
  virtual ~IMetricCollector() = default;

 public:

  virtual MetricActionHandle getHandle(StringView name) =0;
  virtual MetricId beginAction(const MetricActionHandle& handle) =0;
  virtual void endAction(const MetricId& handle) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MetricSentry::
MetricSentry(const MetricActionHandle& handle)
 : m_collector(handle.m_collector)
{
  if (m_collector)
    m_id = m_collector->beginAction(handle);
}

MetricSentry::
~MetricSentry() noexcept(false)
{
  if (m_collector)
    m_collector->endAction(m_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
