// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* TimeMetric.h                                                (C) 2000-2020 */
/*                                                                           */
/* Classes gérant les métriques temporelles.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_TIMEMETRICCOLLECTOR_H
#define ARCCORE_TRACE_TIMEMETRICCOLLECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceGlobal.h"
#include "arccore/trace/ITimeMetricCollector.h"
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

class ARCCORE_BASE_EXPORT TimeMetricActionHandleBuildInfo
{
 public:
  TimeMetricActionHandleBuildInfo(const String& name)
  : m_name(name){}
 public:
  const String& name() const { return m_name; }
 public:
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_BASE_EXPORT TimeMetricActionHandle
{
 public:
  TimeMetricActionHandle(ITimeMetricCollector* c,const String& name)
  : m_collector(c), m_name(name){}
 public:
  ITimeMetricCollector* collector() const { return m_collector; }
  const String& name() const { return m_name; }
 public:
  ITimeMetricCollector* m_collector;
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: n'autoriser que la sémantique std::move
class ARCCORE_BASE_EXPORT TimeMetricId
{
 public:
  TimeMetricId() : m_handle(nullptr){}
  TimeMetricId(const TimeMetricActionHandle* handle) : m_handle(handle){}
  const TimeMetricActionHandle* handlePointer() const { return m_handle; }
 public:
  const TimeMetricActionHandle* m_handle = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_BASE_EXPORT TimeMetricSentry
{
 public:
  inline TimeMetricSentry(const TimeMetricActionHandle& handle);
  inline ~TimeMetricSentry() noexcept(false);
 private:
  ITimeMetricCollector* m_collector;
  TimeMetricId m_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeMetricSentry::
TimeMetricSentry(const TimeMetricActionHandle& handle)
 : m_collector(handle.m_collector)
{
  if (m_collector)
    m_id = m_collector->beginAction(handle);
}

TimeMetricSentry::
~TimeMetricSentry() noexcept(false)
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
