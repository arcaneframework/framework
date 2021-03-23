// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeStats.h                                                 (C) 2000-2021 */
/*                                                                           */
/* Statistiques sur les temps d'exécution.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_TIMESTATS_H
#define ARCANE_IMPL_TIMESTATS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/ITimeStats.h"

#include <stack>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques sur les temps d'exécution.
 */
class TimeStats
: public TraceAccessor
, public ITimeStats
{
 public:

  class Action;
  typedef List<Action*> ActionList;
  class ActionSeries;
  class PhaseValue;
  typedef List<PhaseValue*> PhaseValueList;
  class MetricCollector;

  enum eTimeType
  {
    TT_Real = 0,
    TT_Virtual
  };

  //! Nombre de valeurs de eTimeType
  static const Integer NB_TIME_TYPE = 2;

  static const Integer TC_Local = 0;
  static const Integer TC_Cumulative = 1;

  struct TimeValue
  {
   public:
    TimeValue(Real real_time,Real virtual_time)
    {
      _initTimes();
      m_time[TT_Real][TC_Local] = real_time;
      m_time[TT_Virtual][TC_Local] = virtual_time;
    }
    TimeValue()
    {
      _initTimes();
    }

   public:
    void add(const TimeValue& phase)
    {
      for( Integer i=0; i<NB_TIME_TYPE; ++i )
        m_time[i][TC_Local] += phase.m_time[i][TC_Local];
    }
   public:

    Real m_time[NB_TIME_TYPE][2];
    void _initTimes()
    {
      for( Integer tt=0; tt<NB_TIME_TYPE; ++tt ){
        m_time[tt][TC_Local] = 0.;
        m_time[tt][TC_Cumulative] = 0.;
      }
    }
  };

  class PhaseValue
  : public TimeValue
  {
   public:
    PhaseValue(eTimePhase pt,Real real_time,Real virtual_time)
    : TimeValue(real_time,virtual_time), m_type(pt) { }
    PhaseValue()
    : m_type(TP_Computation) { }
   public:
    eTimePhase m_type;
  };


 public:

  TimeStats(ITimerMng* timer_mng,ITraceMng* trm,const String& name);
  TimeStats(const TimeStats& rhs) = delete;
  TimeStats& operator=(const TimeStats& rhs) = delete;
  ~TimeStats() override;

 public:

  void beginGatherStats() override;
  void endGatherStats() override;

 public:

  void beginAction(const String& action_name) override;
  void endAction(const String& action_name,bool print_time) override;
  void beginPhase(eTimePhase phase_type) override;
  void endPhase(eTimePhase phase_type) override;

 public:
  
  Real elapsedTime(eTimePhase phase) override;
  Real elapsedTime(eTimePhase phase,const String& action) override;

 public:

  void dumpStats(ostream& ostr,bool is_verbose,Real nb,
                 const String& name,bool use_elapsed_time) override;
  void dumpCurrentStats(const String& action) override;

  void dumpTimeAndMemoryUsage(IParallelMng* pm) override;

  bool isGathering() const override;

  void dumpStatsJSON(JSONWriter& writer) override;

  ITimeMetricCollector* metricCollector() override;

  void notifyNewIterationLoop() override;
  void saveTimeValues(Properties* p) override;
  void mergeTimeValues(Properties* p) override;

 private:

  ITimerMng* m_timer_mng = nullptr;
  Timer* m_virtual_timer = nullptr;
  Timer* m_real_timer = nullptr;
  bool m_is_gathering;
  PhaseValue m_current_phase;
  //! Statistiques sur l'exéctution en cours
  ActionSeries* m_current_action_series = nullptr;
  //! Statistiques sur les exécutions précédentes
  ActionSeries* m_previous_action_series = nullptr;
  Action* m_main_action = nullptr;
  Action* m_current_action = nullptr;
  std::stack<eTimePhase> m_phases_type;
  bool m_need_compute_elapsed_time = false;
  ostringstream m_full_stats_str;
  bool m_full_stats;
  String m_name;
  ITimeMetricCollector* m_metric_collector = nullptr;

 private:

  Action* _currentAction();
  PhaseValue _currentPhaseValue();
  void _checkGathering();
  void _computeCumulativeTimes();
  void _dumpCumulativeTime(ostream& ostr,Action& action,eTimePhase tp,eTimeType tt);
  void _dumpAllPhases(ostream& ostr,Action& action,eTimeType tt,int tc,Real nb);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

