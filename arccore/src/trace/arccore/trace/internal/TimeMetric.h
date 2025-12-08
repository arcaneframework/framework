// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeMetric.h                                                (C) 2000-2025 */
/*                                                                           */
/* Classes gérant les métriques temporelles.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_INTERNAL_TIMEMETRIC_H
#define ARCCORE_TRACE_INTERNAL_TIMEMETRIC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceGlobal.h"
#include "arccore/trace/internal/ITimeMetricCollector.h"
#include "arccore/base/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * API en cours de définition. Ne pas utiliser en dehors de Arccore/Arcane.
 */
namespace Arcane
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

class ARCCORE_TRACE_EXPORT TimeMetricActionBuildInfo
{
 public:
  explicit TimeMetricActionBuildInfo(const String& name)
  : m_name(name), m_phase(-1){}
  TimeMetricActionBuildInfo(const String& name,int phase)
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

class ARCCORE_TRACE_EXPORT TimeMetricAction
{
 public:
  TimeMetricAction()
  : m_collector(nullptr), m_phase(-1) {}
  TimeMetricAction(ITimeMetricCollector* c,const TimeMetricActionBuildInfo& x)
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
class ARCCORE_TRACE_EXPORT TimeMetricId
{
 public:
  TimeMetricId() : m_id(-1){}
  explicit TimeMetricId(const TimeMetricAction& action,Int64 id)
  : m_action(action), m_id(id){}
  const TimeMetricAction& action() const { return m_action; }
  Int64 id() const { return m_id; }
 public:
  TimeMetricAction m_action;
  Int64 m_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sentinelle pour collecter les informations temporelles.
 */
class ARCCORE_TRACE_EXPORT TimeMetricSentry
{
 public:
  TimeMetricSentry() : m_collector(nullptr){}
  TimeMetricSentry(TimeMetricSentry&& rhs)
  : m_collector(rhs.m_collector), m_id(rhs.m_id)
  {
    // Met à nul \a rhs pour ne pas qu'il appelle 'endAction'.
    rhs.m_collector = nullptr;
  }
  explicit TimeMetricSentry(const TimeMetricAction& action)
  : m_collector(action.collector())
  {
    if (m_collector)
      m_id = m_collector->beginAction(action);
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
 * \brief s standards pour les phases temporelles.
 */
class ARCCORE_TRACE_EXPORT StandardPhaseTimeMetrics
{
 public:
  StandardPhaseTimeMetrics() = default;
  StandardPhaseTimeMetrics(ITimeMetricCollector* c) { initialize(c); }
 public:
  void initialize(ITimeMetricCollector* collector);
 public:
  //! Action pour indiquer qu'on est dans une phase d'échange de message
  const TimeMetricAction& messagePassingPhase() const { return m_message_passing_phase; }
  //! Action pour indiquer qu'on est dans une phase d'entrée-sortie.
  const TimeMetricAction& inputOutputPhase() const { return m_input_output_phase; }
  //! Action pour indiquer qu'on est dans une phase de calcul.
  const TimeMetricAction& computationPhase() const { return m_computation_phase; }
 private:
  TimeMetricAction m_message_passing_phase;
  TimeMetricAction m_input_output_phase;
  TimeMetricAction m_computation_phase;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_TRACE_EXPORT TimeMetricAction
timeMetricPhaseMessagePassing(ITimeMetricCollector* c);
extern "C++" ARCCORE_TRACE_EXPORT TimeMetricAction
timeMetricPhaseInputOutput(ITimeMetricCollector* c);
extern "C++" ARCCORE_TRACE_EXPORT TimeMetricAction
timeMetricPhaseComputation(ITimeMetricCollector* c);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
