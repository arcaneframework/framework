// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeStats.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Statistiques sur les temps d'exécution.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Deleter.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NameComparer.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/Exception.h"

#include "arcane/core/Timer.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ITimerMng.h"
#include "arcane/core/MathUtils.h"
#include "arcane/core/Properties.h"

#include "arcane/impl/TimeStats.h"

#include "arccore/trace/internal/ITimeMetricCollector.h"
#include "arccore/trace/internal/TimeMetric.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ITimeStats*
arcaneCreateTimeStats(ITimerMng* timer_mng, ITraceMng* trace_mng, const String& name)
{
  return new TimeStats(timer_mng, trace_mng, name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TimeStats::MetricCollector
: public ITimeMetricCollector
{
 public:
  explicit MetricCollector(ITimeStats* ts)
  : m_time_stats(ts)
  , m_id(1)
  {}

 public:
  TimeMetricAction getAction(const TimeMetricActionBuildInfo& x) override
  {
    return { this, x };
  }
  TimeMetricId beginAction(const TimeMetricAction& handle) override
  {
    auto name = handle.name();
    if (!name.null())
      m_time_stats->beginAction(name);
    int phase = handle.phase();
    if (phase >= 0)
      m_time_stats->beginPhase(static_cast<eTimePhase>(phase));
    return TimeMetricId(handle, ++m_id);
  }
  void endAction(const TimeMetricId& metric_id) override
  {
    const TimeMetricAction& action = metric_id.action();
    int phase = action.phase();
    if (phase >= 0)
      m_time_stats->endPhase(static_cast<eTimePhase>(phase));
    auto name = action.name();
    if (!name.null())
      m_time_stats->endAction(name, false);
  }

 private:
  ITimeStats* m_time_stats;
  std::atomic<Int64> m_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Action
 */
class TimeStats::Action
{
 public:
  /*!
   * \brief Informations pour sauver/reconstruire une arborescence d'action
   */
  class AllActionsInfo
  {
   public:
    UniqueArray<String> m_name_list;
    UniqueArray<Int32> m_nb_child;
    UniqueArray<Int64> m_nb_call_list;
    UniqueArray<Real> m_time_list;
    Int64 m_nb_iteration_loop = 0;
   public:
    friend std::ostream& operator<<(std::ostream& o,const AllActionsInfo& x)
    {
      o << "NbLoop=" << x.m_nb_iteration_loop << "\n";
      o << "Name=" << x.m_name_list << "\n";
      o << "NbCall=" << x.m_nb_call_list << "\n";
      o << "NbChild=" << x.m_nb_child << "\n";
      o << "Time=" << x.m_time_list << "\n";
      return o;
    }
   public:
    void clear()
    {
      m_nb_iteration_loop = 0;
      m_name_list.clear();
      m_nb_child.clear();
      m_nb_call_list.clear();
      m_time_list.clear();
    }
  };
 public:
  Action(Action* parent,const String& name)
  : m_parent(parent), m_name(name), m_nb_called(0) {}
  ~Action();
 public:
  //! Action fille de nom \a name. nullptr si aucune avec ce nom
  Action* subAction(const String& name);
  const String& name() const { return m_name; }
  Action* parent() const { return m_parent; }
  Int64 nbCalled() const { return m_nb_called; }
  Action* findOrCreateSubAction(const String& name);
  void addPhaseValue(const PhaseValue& new_pv);
  void addNbCalled() { ++m_nb_called; }
  Action* findSubActionRecursive(const String& action_name) const;
  void save(AllActionsInfo& save_info) const;
  void merge(AllActionsInfo& save_info,Integer* index);
  void dumpJSON(JSONWriter& writer,eTimeType tt);
  void computeCumulativeTimes();
  void dumpCurrentStats(std::ostream& ostr,int level,Real unit);
  void reset();
 private:
  Action* m_parent; //!< Action parente
  String m_name; //!< Nom de l'action
  Int64 m_nb_called; //!< Nombre de fois que l'action a été appelée
 private:
  void _addSubAction(Action* sub) { m_sub_actions.add(sub); }
 public:
  ActionList m_sub_actions; //!< Actions filles
  PhaseValue m_phases[NB_TIME_PHASE];
  /*
   * Cette valeur est calculée par computeCumulativeTimes() et ne doit
   * pas être conservée.
   */
  TimeValue m_total_time;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Série d'actions.
 */
class TimeStats::ActionSeries
{
  using AllActionsInfo = TimeStats::Action::AllActionsInfo;
 public:
  ActionSeries()
  : m_main_action(nullptr,"Main")
  {
  }
  //! Créé une série qui cumule les temps des deux séries passées en argument
  ActionSeries(const ActionSeries& s1,const ActionSeries& s2)
  : m_main_action(nullptr,"Main")
  {
    Action::AllActionsInfo action_info;
    s1.save(action_info);
    this->merge(action_info);
    s2.save(action_info);
    this->merge(action_info);
  }
  ~ActionSeries()
  {
  }
 public:
  Action* mainAction() { return &m_main_action; }
  Int64 nbIterationLoop() const { return m_nb_iteration_loop; }
  void save(AllActionsInfo& all_actions_info) const
  {
    all_actions_info.clear();
    m_main_action.save(all_actions_info);
    all_actions_info.m_nb_iteration_loop = m_nb_iteration_loop;
  }
  void merge(AllActionsInfo& all_actions_info)
  {
    m_nb_iteration_loop += all_actions_info.m_nb_iteration_loop;
    Integer index = 0;
    m_main_action.merge(all_actions_info,&index);
    m_main_action.computeCumulativeTimes();
  }
  void dumpStats(std::ostream& ostr,bool is_verbose,Real nb,const String& name,
                 bool use_elapsed_time,const String& message);
 public:

  Action m_main_action;
  Int64 m_nb_iteration_loop = 0;

 private:
  void _dumpStats(std::ostream& ostr,Action& action,eTimeType tt,int level,int max_level,Real nb);
  void _dumpAllPhases(std::ostream& ostr,Action& action,eTimeType tt,int tc,Real nb);
  void _dumpCurrentStats(std::ostream& ostr,Action& action,int level,Real unit);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeStats::
TimeStats(ITimerMng* timer_mng, ITraceMng* trace_mng, const String& name)
: TraceAccessor(trace_mng)
, m_timer_mng(timer_mng)
, m_virtual_timer(nullptr)
, m_real_timer(nullptr)
, m_is_gathering(false)
, m_current_action_series(new ActionSeries())
, m_previous_action_series(new ActionSeries())
, m_main_action(m_current_action_series->mainAction())
, m_current_action(m_main_action)
, m_need_compute_elapsed_time(true)
, m_full_stats(false)
, m_name(name)
, m_metric_collector(new MetricCollector(this))
{
  m_phases_type.push(TP_Computation);
  if (platform::getEnvironmentVariable("ARCANE_FULLSTATS") == "TRUE")
    m_full_stats = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeStats::
~TimeStats()
{
  delete m_metric_collector;
  if (m_is_gathering)
    arcaneCallFunctionAndCatchException([&]() { endGatherStats(); });
  delete m_virtual_timer;
  delete m_real_timer;
  delete m_previous_action_series;
  delete m_current_action_series;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
beginGatherStats()
{
  if (m_is_gathering)
    ARCANE_FATAL("Already gathering");

  if (!m_virtual_timer)
    m_virtual_timer = new Timer(m_timer_mng, "SubDomainVirtual", Timer::TimerVirtual);
  if (!m_real_timer)
    m_real_timer = new Timer(m_timer_mng, "SubDomainReal", Timer::TimerReal);

  m_is_gathering = true;

  m_current_phase = PhaseValue();

  m_virtual_timer->start();
  m_real_timer->start();
  m_full_stats_str << "<? xml version='1.0'?>\n";
  m_full_stats_str << "<stats>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
endGatherStats()
{
  m_virtual_timer->stop();
  m_real_timer->stop();

  m_is_gathering = false;
  if (m_full_stats) {
    m_full_stats_str << "</stats>\n";
    StringBuilder sb = "stats-";
    sb += m_name;
    sb += ".xml";
    String s(sb);
    std::ofstream ofile(s.localstr());
    ofile << m_full_stats_str.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeStats::Action* TimeStats::Action::
findOrCreateSubAction(const String& name)
{
  Action* sa = subAction(name);
  if (!sa){
    sa = new Action(this,name);
    _addSubAction(sa);
  }
  return sa;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
beginAction(const String& action_name)
{
  _checkGathering();
  Action* current_action = _currentAction();
  current_action->addPhaseValue(_currentPhaseValue());
  Action* sa = current_action->findOrCreateSubAction(action_name);
  if (m_full_stats)
    m_full_stats_str << "<action name='" << sa->name() << "'"
                     << ">\n";
  m_current_action = sa;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
endAction(const String& action_name,bool print_time)
{
  ARCANE_UNUSED(action_name);
  _checkGathering();
  m_need_compute_elapsed_time = true;
  TimeStats::PhaseValue pv = _currentPhaseValue();
  m_current_action->addPhaseValue(pv);
  m_current_action->addNbCalled();
  if (print_time){
    elapsedTime(TP_Computation,m_current_action->name());
    elapsedTime(TP_Communication,m_current_action->name());
  }
  if (m_full_stats)
    m_full_stats_str << "</action><!-- "  << m_current_action->name() << " -->\n";
  m_current_action = m_current_action->parent();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
beginPhase(eTimePhase phase_type)
{
  _checkGathering();
  TimeStats::PhaseValue pv = _currentPhaseValue();
  m_current_action->addPhaseValue(pv);
  m_current_phase.m_type = phase_type;
  m_phases_type.push(pv.m_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
endPhase(eTimePhase phase_type)
{
  ARCANE_UNUSED(phase_type);
  _checkGathering();
  m_need_compute_elapsed_time = true;
  TimeStats::PhaseValue pv = _currentPhaseValue();
  m_current_action->addPhaseValue(pv);
  if (m_phases_type.empty())
    ARCANE_FATAL("No previous phases");
  eTimePhase old_phase_type = m_phases_type.top();
  m_phases_type.pop();
  m_current_phase.m_type = old_phase_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimeStats::
elapsedTime(eTimePhase phase)
{
  _computeCumulativeTimes();
  return m_main_action->m_phases[phase].m_time[TT_Real][TC_Cumulative];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimeStats::
elapsedTime(eTimePhase phase,const String& action_name)
{
  _computeCumulativeTimes();
  Action* action = m_main_action->findSubActionRecursive(action_name);
  if (!action)
    return 0.0;
  info() << "TimeStat: type=" << phase << " action=" << action_name
         << " local_Real=" << action->m_phases[phase].m_time[TT_Real][TC_Local]
         << " total_Real=" << action->m_phases[phase].m_time[TT_Real][TC_Cumulative]
         << " local_Virt=" << action->m_phases[phase].m_time[TT_Virtual][TC_Local]
         << " total_Virt=" << action->m_phases[phase].m_time[TT_Virtual][TC_Cumulative];
  return action->m_phases[phase].m_time[TT_Real][TC_Cumulative];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeStats::Action* TimeStats::Action::
findSubActionRecursive(const String& action_name) const
{
  for( ActionList::Enumerator i(this->m_sub_actions); ++i; ){
    Action* action = *i;
    if (action->name()==action_name)
      return action;
    Action* find_action = action->findSubActionRecursive(action_name);
    if (find_action)
      return find_action;
  }
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::ActionSeries::
dumpStats(std::ostream& ostr,bool is_verbose,Real nb,const String& name,
          bool use_elapsed_time,const String& message)
{
  Int64 nb_iteration_loop = this->nbIterationLoop();
  if (nb_iteration_loop!=0)
    nb = nb * ((Real)nb_iteration_loop);
  eTimeType tt = TT_Virtual;
  if (use_elapsed_time)
    tt = TT_Real;
  ostr << "-- Execution statistics " << message
       << " (divide=" << nb << ", nb_loop=" << nb_iteration_loop << ")";
  if (tt==TT_Real)
    ostr << " (clock time)";
  else if (tt==TT_Virtual)
    ostr << " (CPU time)";
  ostr << ":\n";
  std::ios_base::fmtflags f = ostr.flags(std::ios::right);

  ostr << Trace::Width(50) << "        Action       "
       << Trace::Width(11) << "  Time  "
       << Trace::Width(11) << "  Time  "
       << Trace::Width(8) << "N"
       << '\n';
  ostr << Trace::Width(50) << " "
       << Trace::Width(11) << "Total(s)"
       << Trace::Width(11) << (String("/")+name+"(us)")
       << '\n';
  ostr << '\n';
  if (is_verbose){
    _dumpStats(ostr,m_main_action,tt,1,0,nb);
  }
  else{
    // Affiche seulement les statistiques concernant les temps
    // pour chaque module.
    Action* action = m_main_action.findSubActionRecursive("Loop");
    if (!action)
      _dumpStats(ostr,m_main_action,tt,1,3,nb);
    else
      _dumpStats(ostr,*action,tt,1,3,nb);
  }
  ostr.flags(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
dumpStats(std::ostream& ostr,bool is_verbose,Real nb,const String& name,
          bool use_elapsed_time)
{
  _computeCumulativeTimes();
  ostr << "Execution statistics (current execution)\n";
  m_current_action_series->dumpStats(ostr,is_verbose,nb,name,use_elapsed_time,"(current execution)");
  // N'affiche les statistiques cumulatives que s'il y a déjà eu une éxecution.
  if (m_previous_action_series->nbIterationLoop()!=0){
    ostr << "\nExecution statistics (cumulative)\n";
    ActionSeries cumul_series(*m_previous_action_series,*m_current_action_series);
    cumul_series.dumpStats(ostr,is_verbose,nb,name,use_elapsed_time,"(cumulative execution)");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
dumpCurrentStats(const String& action_name)
{
  Action* action = m_main_action->findSubActionRecursive(action_name);
  if (!action)
    return;
  _computeCumulativeTimes();
  Real unit = 1.e3;
  OStringStream ostr;
  action->dumpCurrentStats(ostr(),1,unit);
  info() << "-- Execution statistics: Action=" << action->name()
         << "\n"
         << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
resetStats(const String& action_name)
{
  Action* action = m_main_action->findSubActionRecursive(action_name);
  if (!action)
    return;
  action->reset();
  m_need_compute_elapsed_time = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
void
_writeValue(std::ostream& ostr,Real value,Real unit)
{
  ostr.width(12);
  Real v2 = value * unit;
  Integer i_unit = Convert::toInteger(unit);
  if (i_unit==0)
    i_unit = 1;
  Integer i_v2 = Convert::toInteger(v2);
  ostr << i_v2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
_printIndentedName(std::ostream& ostr,const String& name,int level)
{
  StringBuilder indent_str;
  StringBuilder after_str;
  for( int i=0; i<level; ++i )
    indent_str.append(" ");
  ostr << indent_str;
  ostr << name;
  int alen = static_cast<int>(name.utf8().size());
  alen += level;
  for( int i=0; i<50-alen; ++i )
    after_str += " ";
  ostr << after_str;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
_printPercentage(std::ostream& ostr,Real value,Real cumulative_value)
{
  Real percent = 1.0;
  // Normalement il faut juste vérifier que cumulative_value n'est pas nul
  // pour faire la division. Cependant, plusieurs compilateurs (icc sur ia64,
  // clang 3.7.0) semblent un peu agressif au niveau de spéculations
  // (avec -O2) et font la division même si le test est faux ce qui
  // provoque un SIGFPE. Pour contourner cela, il semble que faire
  // deux comparaisons fonctionne.
  Real z_cumulative_value = cumulative_value;
  if (z_cumulative_value!=0.0 && !math::isNearlyZero(z_cumulative_value)){
    percent = value / z_cumulative_value;
  }
  percent *= 1000.0;
  Integer n_percent = Convert::toInteger(percent);
  ostr.width(3);
  ostr << (n_percent / 10) << '.' << (n_percent % 10);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::Action::
dumpCurrentStats(std::ostream& ostr,int level,Real unit)
{
  Action& action = *this;
  _printIndentedName(ostr,action.name(),level);
  _writeValue(ostr,action.m_phases[TP_Computation].m_time[TT_Real][TC_Cumulative],unit);
  _writeValue(ostr,action.m_phases[TP_Communication].m_time[TT_Real][TC_Cumulative],unit);
  ostr << '\n';
  for( ActionList::Enumerator i(action.m_sub_actions); ++i; ){
    Action* a = *i;
    a->dumpCurrentStats(ostr,level+1,unit);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
_computeCumulativeTimes()
{
  if (!m_need_compute_elapsed_time)
    return;
  m_main_action->computeCumulativeTimes();
  m_need_compute_elapsed_time = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::ActionSeries::
_dumpStats(std::ostream& ostr,Action& action,eTimeType tt,int level,int max_level,Real nb)
{
  PhaseValue& pv = action.m_phases[TP_Computation];
  _printIndentedName(ostr,action.name(),level);

  if (pv.m_time[TT_Real][TC_Cumulative]!=0.0 || pv.m_time[TT_Virtual][TC_Cumulative]!=0.0){
    _dumpAllPhases(ostr,action,tt,TC_Cumulative,nb);
  }
  ostr << '\n';
  if (max_level==0 || level<max_level){
    for( ActionList::Enumerator i(action.m_sub_actions); ++i; ){
      Action* a = *i;
      _dumpStats(ostr,*a,tt,level+1,max_level,nb);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
_dumpCumulativeTime(std::ostream& ostr,Action& action,eTimePhase tp,eTimeType tt)
{
  Real current_time = action.m_phases[tp].m_time[tt][TC_Local];
  Real cumulative_time = action.m_phases[tp].m_time[tt][TC_Cumulative];

  ostr.width(12);
  ostr << current_time << ' ';
  ostr.width(12);
  ostr << cumulative_time << ' ';

  _printPercentage(ostr,current_time,m_main_action->m_phases[tp].m_time[tt][TC_Cumulative]);
  _printPercentage(ostr,cumulative_time,m_main_action->m_phases[tp].m_time[tt][TC_Cumulative]);
  {
    Action* parent_action = action.parent();
    Real parent_time = cumulative_time;
    if (parent_action)
      parent_time = parent_action->m_phases[tp].m_time[tt][TC_Cumulative];
    _printPercentage(ostr,cumulative_time,parent_time);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::ActionSeries::
_dumpAllPhases(std::ostream& ostr,Action& action,eTimeType tt,int tc,Real nb)
{
  Real all_phase_time = action.m_total_time.m_time[tt][tc];

  // Temps passé dans l'action
  ostr << Trace::Width(11) << String::fromNumber(all_phase_time,3);

  // Temps passé dans l'action par \a nb
  // Si nb vaut 0, prend le nombre d'appel
  {
    Real ct_by_call = 0;
    Real nb_called = nb;
    if (math::isZero(nb_called))
      nb_called = (Real)action.nbCalled();
    if (!math::isZero(nb_called)){
      Real r = all_phase_time * 1.0e6;
      Real r_nb_called = static_cast<Real>(nb_called);
      // Ajoute un epsilon pour éviter une exécution spéculative si \a nb_called vaut 0.
      ct_by_call = r / (r_nb_called + 1.0e-10);
    }
    ostr << Trace::Width(11) << String::fromNumber(ct_by_call,3);
  }

  // Nombre d'appel
  ostr.width(9);
  ostr << action.nbCalled() << ' ';

  _printPercentage(ostr,all_phase_time,m_main_action.m_total_time.m_time[tt][tc]);
  ostr << ' ';
  {
    Action* parent_action = action.parent();
    Real parent_time = all_phase_time;
    if (parent_action)
      parent_time = parent_action->m_total_time.m_time[tt][tc];
    _printPercentage(ostr,all_phase_time,parent_time);
    ostr << ' ';
  }

  ostr << "[";
  for( Integer phase=0; phase<NB_TIME_PHASE; ++phase ){
    _printPercentage(ostr,action.m_phases[phase].m_time[tt][tc],all_phase_time);
    if ((phase+1)!=NB_TIME_PHASE)
      ostr << ' ';
  }
  ostr << "]";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::Action::
computeCumulativeTimes()
{
  Action& action = *this;
  for( Integer tt=0; tt<NB_TIME_TYPE; ++tt ){
    action.m_total_time.m_time[tt][TC_Local] = 0.;
    action.m_total_time.m_time[tt][TC_Cumulative] = 0.;
    for( Integer phase=0; phase<NB_TIME_PHASE; ++phase ){
      Real r = action.m_phases[phase].m_time[tt][TC_Local];
      action.m_phases[phase].m_time[tt][TC_Cumulative] = r;
      action.m_total_time.m_time[tt][TC_Cumulative] += r;
      action.m_total_time.m_time[tt][TC_Local] += r;
    }
  }

  for( ActionList::Enumerator i(action.m_sub_actions); ++i; ){
    Action* a = *i;
    a->computeCumulativeTimes();
    for( Integer phase=0; phase<NB_TIME_PHASE; ++phase ){
      for( Integer tt=0; tt<NB_TIME_TYPE; ++tt ){
        Real r = a->m_phases[phase].m_time[tt][TC_Cumulative];
        action.m_phases[phase].m_time[tt][TC_Cumulative] += r;
        action.m_total_time.m_time[tt][TC_Cumulative] += r;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeStats::Action* TimeStats::
_currentAction()
{
  if (!m_current_action)
    m_current_action = m_main_action;
  return m_current_action;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeStats::PhaseValue TimeStats::
_currentPhaseValue()
{
  Real real_time = m_timer_mng->getTime(m_real_timer);
  Real virtual_time = m_timer_mng->getTime(m_virtual_timer);

  Real diff_real_time = real_time - m_current_phase.m_time[TT_Real][TC_Local];
  Real diff_virtual_time = virtual_time - m_current_phase.m_time[TT_Virtual][TC_Local];
  if (diff_real_time<0.0 || diff_virtual_time<0.0)
    info() << "BAD_CURRENT_PHASE_VALUE " << diff_real_time << " " << diff_virtual_time
           << " phase=" << m_current_phase.m_type;
  m_current_phase.m_time[TT_Real][TC_Local] = real_time;
  m_current_phase.m_time[TT_Virtual][TC_Local] = virtual_time;
  if (m_full_stats)
    m_full_stats_str << "<time"
                     << " phase='" << m_current_phase.m_type << "'"
                     << " real_time='" << real_time << "'"
                     << "/>\n";
  return PhaseValue(m_current_phase.m_type,diff_real_time,diff_virtual_time);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
_checkGathering()
{
  if (!m_is_gathering)
    ARCANE_FATAL("TimeStats::beginGatherStats() not called");
  if (!m_current_action)
    ARCANE_FATAL("No current action");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TimeStats::
isGathering() const
{
  bool is_gather = m_is_gathering && m_current_action;
  return is_gather;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
dumpTimeAndMemoryUsage(IParallelMng* pm)
{
  MessagePassing::dumpDateAndMemoryUsage(pm, traceMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
dumpStatsJSON(JSONWriter& writer)
{
  _computeCumulativeTimes();
  writer.write("Version",(Int64)1);

  writer.writeKey("Current");
  writer.beginObject();
  m_main_action->dumpJSON(writer,TT_Real);
  writer.endObject();

  // Affiche les statistiques cumulatives que s'il y a déjà eu une éxecution.
  if (m_previous_action_series->nbIterationLoop()!=0){
    ActionSeries cumul_series(*m_previous_action_series,*m_current_action_series);
    writer.writeKey("Cumulative");
    writer.beginObject();
    cumul_series.mainAction()->dumpJSON(writer,TT_Real);
    writer.endObject();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::Action::
dumpJSON(JSONWriter& writer,eTimeType tt)
{
  Action& action = *this;
  writer.writeKey(action.name());
  writer.beginObject();
  
  Real values[NB_TIME_PHASE];
  for( Integer phase=0; phase<NB_TIME_PHASE; ++phase )
    values[phase] = action.m_phases[phase].m_time[tt][TC_Local];    
  writer.write("Local",RealArrayView(NB_TIME_PHASE,values));
  for( Integer phase=0; phase<NB_TIME_PHASE; ++phase )
    values[phase] = action.m_phases[phase].m_time[tt][TC_Cumulative];    
  writer.write("Cumulative",RealArrayView(NB_TIME_PHASE,values));

  if (!action.m_sub_actions.empty()){
    writer.writeKey("SubActions");
    writer.beginArray();
    for( ActionList::Enumerator i(action.m_sub_actions); ++i; ){
      Action* a = *i;
      a->dumpJSON(writer,tt);
    }
    writer.endArray();
  }

  writer.endObject();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeStats::Action::
~Action()
{
  m_sub_actions.each(Deleter());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeStats::Action* TimeStats::Action::
subAction(const String& name)
{
  ActionList::iterator i = m_sub_actions.find_if(NameComparer(name));
  if (i!=m_sub_actions.end())
    return *i;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::Action::
addPhaseValue(const PhaseValue& new_pv)
{
  m_phases[new_pv.m_type].add(new_pv);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::Action::
save(AllActionsInfo& save_info) const
{
  save_info.m_name_list.add(m_name);
  save_info.m_nb_call_list.add(m_nb_called);
  save_info.m_nb_child.add(m_sub_actions.count());
  for( Integer phase=0; phase<NB_TIME_PHASE; ++phase )
    for( Integer i=0; i<NB_TIME_TYPE; ++i )
      save_info.m_time_list.add(m_phases[phase].m_time[i][TC_Local]);
  for( Action* s : m_sub_actions )
    s->save(save_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::Action::
merge(AllActionsInfo& save_info,Integer* index_ptr)
{
  Integer index = *index_ptr;
  String saved_name = save_info.m_name_list[index];
  if (saved_name!=m_name)
    ARCANE_FATAL("Bad merge name={0} saved={1}",m_name,saved_name);
  ++(*index_ptr);
  Integer nb_child = save_info.m_nb_child[index];
  m_nb_called += save_info.m_nb_call_list[index];
  {
    Integer pos = index * (NB_TIME_PHASE*NB_TIME_TYPE);
    for (Integer phase = 0; phase < NB_TIME_PHASE; ++phase)
      for (Integer i = 0; i < NB_TIME_TYPE; ++i) {
        m_phases[phase].m_time[i][TC_Local] += save_info.m_time_list[pos];
        ++pos;
      }
  }
  for( Integer i=0; i<nb_child; ++i ) {
    String next_name = save_info.m_name_list[*index_ptr];
    Action* a = findOrCreateSubAction(next_name);
    a->merge(save_info,index_ptr);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remet à zéro les statistiques de l'action et de ces filles.
 */
void TimeStats::Action::
reset()
{
  m_nb_called = 0;
  for( Integer phase=0; phase<NB_TIME_PHASE; ++phase )
    for (Integer i = 0; i < NB_TIME_TYPE; ++i)
      m_phases[phase].m_time[i][TC_Local] = 0.0;
      
  for( Action* s : m_sub_actions )
    s->reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimeMetricCollector* TimeStats::
metricCollector()
{
  return m_metric_collector;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
saveTimeValues(Properties* p)
{
  info(4) << "Saving TimeStats values";
  Action::AllActionsInfo action_save_info;
  ActionSeries cumulative_series(*m_previous_action_series,*m_current_action_series);
  cumulative_series.save(action_save_info);
  const bool is_verbose = false;
  if (is_verbose) {
    info() << "Saved " << action_save_info;
  }

  p->set("Version",1);
  p->set("NbIterationLoop",action_save_info.m_nb_iteration_loop);
  p->set("Names",action_save_info.m_name_list);
  p->set("NbCalls",action_save_info.m_nb_call_list);
  p->set("NbChildren",action_save_info.m_nb_child);
  p->set("TimeList",action_save_info.m_time_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
mergeTimeValues(Properties* p)
{
  info(4) << "Merging TimeStats values";

  Action::AllActionsInfo action_save_info;

  Int32 v = p->getInt32WithDefault("Version",0);
  // Ne fait rien si aucune info dans la protection
  if (v==0)
    return;
  if (v!=1){
    info() << "Warning: can not merge time stats values because checkpoint version is not compatible";
    return;
  }

  action_save_info.m_nb_iteration_loop = p->getInt64("NbIterationLoop");
  p->get("Names",action_save_info.m_name_list);
  p->get("NbCalls",action_save_info.m_nb_call_list);
  p->get("NbChildren",action_save_info.m_nb_child);
  p->get("TimeList",action_save_info.m_time_list);

  const bool is_verbose = false;
  if (is_verbose) {
    info() << "MergedSeries=" << action_save_info;
  }
  m_previous_action_series->merge(action_save_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeStats::
notifyNewIterationLoop()
{
  ++m_current_action_series->m_nb_iteration_loop;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
