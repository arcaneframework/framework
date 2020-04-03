// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* TimeMetric.h                                                (C) 2000-2020 */
/*                                                                           */
/* Classes gérant les métriques temporelles.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_TIMEMETRIC_H
#define ARCCORE_TRACE_TIMEMETRIC_H
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
/*!
 * \brief Catégories standards pour les phases temporelles.
 *
 * \note Les valeurs de ces catégories doivent correspondre à celle
 * de l'énumération eTimePhase de %Arcane.
 */
enum class TimeMetricPhase
{
  Computation = 0,
  MessagePassing = 1,
  InputOutput = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_BASE_EXPORT TimeMetricActionHandleBuildInfo
{
 public:
  explicit TimeMetricActionHandleBuildInfo(const String& name)
  : m_name(name), m_phase(-1){}
  TimeMetricActionHandleBuildInfo(const String& name,int phase)
  : m_name(name), m_phase(phase){}
 public:
  const String& name() const { return m_name; }
  int phase() const { return m_phase; }
 public:
  String m_name;
  int m_phase = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_BASE_EXPORT TimeMetricActionHandle
{
 public:
  TimeMetricActionHandle()
  : m_collector(nullptr), m_phase(-1) {}
  TimeMetricActionHandle(ITimeMetricCollector* c,const TimeMetricActionHandleBuildInfo& x)
  : m_collector(c), m_name(x.name()), m_phase(x.phase()){}
 public:
  ITimeMetricCollector* collector() const { return m_collector; }
  const String& name() const { return m_name; }
  int phase() const { return m_phase; }
 private:
  ITimeMetricCollector* m_collector;
  String m_name;
  int m_phase;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: n'autoriser que la sémantique std::move
class ARCCORE_BASE_EXPORT TimeMetricId
{
 public:
  TimeMetricId() : m_id(-1){}
  explicit TimeMetricId(const TimeMetricActionHandle& handle,Int64 id)
  : m_handle(handle), m_id(id){}
  const TimeMetricActionHandle& handle() const { return m_handle; }
  Int64 id() const { return m_id; }
 public:
  TimeMetricActionHandle m_handle;
  Int64 m_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sentinelle pour collecter les informations temporelles.
 */
class ARCCORE_BASE_EXPORT TimeMetricSentry
{
 public:
  TimeMetricSentry() : m_collector(nullptr){}
  TimeMetricSentry(TimeMetricSentry&& rhs)
  : m_collector(rhs.m_collector), m_id(rhs.m_id)
  {
    // Met à nul \a rhs pour ne pas qu'il appelle 'endAction'.
    rhs.m_collector = nullptr;
  }
  explicit TimeMetricSentry(const TimeMetricActionHandle& handle)
  : m_collector(handle.collector())
  {
    if (m_collector)
      m_id = m_collector->beginAction(handle);
  }
  ~TimeMetricSentry() noexcept(false)
  {
  if (m_collector)
    m_collector->endAction(m_id);
  }
 private:
  ITimeMetricCollector* m_collector;
  TimeMetricId m_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Handles standards pour les phases temporelles.
 */
class ARCCORE_BASE_EXPORT StandardPhaseTimeMetrics
{
 public:
  StandardPhaseTimeMetrics() = default;
  StandardPhaseTimeMetrics(ITimeMetricCollector* c) { initialize(c); }
 public:
  void initialize(ITimeMetricCollector* collector);
 public:
  //! Handle pour indiquer qu'on est dans une phase d'échange de message
  const TimeMetricActionHandle& messagePassingPhase();
  //! Handle pour indiquer qu'on est dans une phase d'entrée-sortie.
  const TimeMetricActionHandle& inputOutputPhase();
  //! Handle pour indiquer qu'on est dans une phase de calcul.
  const TimeMetricActionHandle& computationPhase();
 private:
  TimeMetricActionHandle m_message_passing_phase;
  TimeMetricActionHandle m_input_output_phase;
  TimeMetricActionHandle m_computation_phase;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
