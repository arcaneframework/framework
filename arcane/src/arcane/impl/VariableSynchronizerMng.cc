// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerMng.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire des synchroniseurs de variables.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableSynchronizerMng.h"

#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableSynchronizerEventArgs.h"
#include "arcane/core/IVariable.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizerStats
: public TraceAccessor
{
 public:

  class StatInfo
  {
   public:

    Int32 m_count = 0;
    Int32 m_nb_same = 0;
    Int32 m_nb_different = 0;
  };

 public:

  explicit VariableSynchronizerStats(ITraceMng* tm)
  : TraceAccessor(tm)
  {}

 public:

  void init(VariableSynchronizerMng* vsm)
  {
    auto handler = [&](const VariableSynchronizerEventArgs& args) {
      _handleEvent(args);
    };
    vsm->onSynchronized().attach(m_observer_pool, handler);
    m_is_event_registered = true;
  }

  void dumpStats(std::ostream& ostr)
  {
    std::streamsize old_precision = ostr.precision(20);
    ostr << "Synchronization Stats\n";
    ostr << Trace::Width(40) << "Variable name"
         << Trace::Width(8) << "Count"
         << Trace::Width(8) << "NbSame"
         << Trace::Width(8) << "NbDiff"
         << "\n";
    for (const auto& p : m_stats) {
      ostr << Trace::Width(40) << p.first
           << " " << Trace::Width(7) << p.second.m_count
           << " " << Trace::Width(7) << p.second.m_nb_same
           << " " << Trace::Width(7) << p.second.m_nb_different
           << "\n";
    }
    ostr.precision(old_precision);
  }

 private:

  EventObserverPool m_observer_pool;
  std::map<String, StatInfo> m_stats;
  bool m_is_event_registered = false;

 private:

  void _handleEvent(const VariableSynchronizerEventArgs& args)
  {
    // On ne traite que les évènements de fin de synchronisation
    if (args.state() != VariableSynchronizerEventArgs::State::EndSynchronize)
      return;
    auto compare_status_list = args.compareStatusList();
    {
      Int32 index = 0;
      for (IVariable* var : args.variables()) {
        auto& v = m_stats[var->fullName()];
        ++v.m_count;
        VariableSynchronizerEventArgs::CompareStatus s = compare_status_list[index];
        if (s == VariableSynchronizerEventArgs::CompareStatus::Same)
          ++v.m_nb_same;
        else if (s == VariableSynchronizerEventArgs::CompareStatus::Different)
          ++v.m_nb_different;
        ++index;
      }
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerMng::
VariableSynchronizerMng(IVariableMng* vm)
: TraceAccessor(vm->traceMng())
, m_variable_mng(vm)
, m_stats(new VariableSynchronizerStats(vm->traceMng()))
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_AUTO_COMPARE_SYNCHRONIZE", true)) {
    m_is_compare_synchronize = (v.value() != 0);
    // Si on active la comparaison, on active aussi les statistiques
    m_is_do_stats = m_is_compare_synchronize;
  }
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_SYNCHRONIZE_STATS", true))
    m_is_do_stats = (v.value() != 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerMng::
~VariableSynchronizerMng()
{
  delete m_stats;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::
initialize()
{
  m_stats->init(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::
dumpStats(std::ostream& ostr) const
{
  m_stats->dumpStats(ostr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
