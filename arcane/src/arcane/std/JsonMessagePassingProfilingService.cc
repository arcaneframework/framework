// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JsonMessagePassingProfilingService.cc                       (C) 2000-2022 */
/*                                                                           */
/* Performance information for "message passing" in JSON format              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/JsonMessagePassingProfilingService.h"

#include <algorithm>
#include <string>

#include "arcane/core/ServiceFactory.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IEntryPoint.h"

#include "arcane/parallel/IStat.h"
#include "arcane/utils/JSONWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(JsonMessagePassingProfilingService,
                        ServiceProperty("JsonMessagePassingProfiling", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMessagePassingProfilingService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JsonMessagePassingProfilingService::
JsonMessagePassingProfilingService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
, m_sub_domain(sbi.subDomain())
, m_json_writer(nullptr)
, m_impl_name("JsonMessagePassingProfiling")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JsonMessagePassingProfilingService::
~JsonMessagePassingProfilingService()
{
  if (m_json_writer)
    delete m_json_writer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
startProfiling()
{
  // creation of the json writer
  if (!m_json_writer) {
    m_json_writer = new JSONWriter(JSONWriter::FormatFlags::None);
    m_json_writer->beginObject();
  }

  // When starting profiling, we begin observing the timeLoopMng events
  m_observer.addObserver(this, &JsonMessagePassingProfilingService::_updateFromBeginEntryPointEvt,
                         m_sub_domain->timeLoopMng()->observable(eTimeLoopEventType::BeginEntryPoint));
  m_observer.addObserver(this, &JsonMessagePassingProfilingService::_updateFromEndEntryPointEvt,
                         m_sub_domain->timeLoopMng()->observable(eTimeLoopEventType::EndEntryPoint));
  m_observer.addObserver(this, &JsonMessagePassingProfilingService::_updateFromBeginIterationEvt,
                         m_sub_domain->timeLoopMng()->observable(eTimeLoopEventType::BeginIteration));
  m_observer.addObserver(this, &JsonMessagePassingProfilingService::_updateFromEndIterationEvt,
                         m_sub_domain->timeLoopMng()->observable(eTimeLoopEventType::EndIteration));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
stopProfiling()
{
  // Stop observations
  m_observer.detachAll();
  // We flush the information
  if (m_json_writer) {
    _dumpCurrentIterationInJSON();
    m_json_writer->endObject();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void JsonMessagePassingProfilingService::
printInfos(std::ostream& output)
{
  output << m_json_writer->getBuffer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
String JsonMessagePassingProfilingService::
implName()
{
  return m_impl_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
_dumpCurrentIterationInJSON()
{
  if (m_ep_mpstat_col.empty())
    return;

  // Iteration number (offset by one because the counter increments before the notification)
  JSONWriter::Object obj_it(*m_json_writer,
                            String::fromNumber(m_sub_domain->commonVariables().globalIteration() - 1));
  // Per entry point
  for (const auto& ep_mpstat : m_ep_mpstat_col) {
    // Alias for the message passing stats for this entry point
    const auto& stats_(ep_mpstat.second.stats());
    // We write the entry point only if there is a non-zero message passing stat
    if (std::any_of(stats_.cbegin(), stats_.cend(), [&](const Arccore::MessagePassing::OneStat& os) { return static_cast<bool>(os.nbMessage()); })) {
      // Entry point name
      JSONWriter::Object obj_ep(*m_json_writer, ep_mpstat.first);
      // Per message passing stat
      for (const auto& mpstat : stats_) {
        // Message passing stats if they exist
        if (mpstat.nbMessage())
          Parallel::dumpJSON(*m_json_writer, mpstat, false);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
_updateFromBeginEntryPointEvt()
{
  // When entering the entry point, we ensure that the stats are reset to zero
  m_sub_domain->parallelMng()->stat()->toArccoreStat()->resetCurrentStat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
_updateFromEndEntryPointEvt()
{
  // We retrieve the entry point name
  const String& ep_name(m_sub_domain->timeLoopMng()->currentEntryPoint()->name());
  auto pos(m_ep_mpstat_col.find(ep_name));
  // If there are no stats yet for this entry point, we create them
  Arccore::MessagePassing::StatData stat_data(m_sub_domain->parallelMng()->stat()->toArccoreStat()->stats());
  if (pos == m_ep_mpstat_col.end())
    m_ep_mpstat_col.emplace(ep_name, stat_data);
  else // otherwise, we merge the stats
    pos->second.mergeAllData(stat_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
_updateFromBeginIterationEvt()
{
  // We start with an empty structure, since it is dumped at the end of each iteration
  m_ep_mpstat_col.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
_updateFromEndIterationEvt()
{
  // We dump everything we have recorded for this iteration
  if (m_json_writer)
    _dumpCurrentIterationInJSON();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
