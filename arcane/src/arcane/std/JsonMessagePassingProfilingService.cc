// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JsonMessagePassingProfilingService.cc                       (C) 2000-2019 */
/*                                                                           */
/* Informations de performances du "message passing" au format JSON          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/JsonMessagePassingProfilingService.h"
#include <algorithm>
#include <string>
#include "arcane/ServiceFactory.h"
#include "arcane/CommonVariables.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IEntryPoint.h"
#include "arcane/parallel/IStat.h"
#include "arcane/utils/JSONWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(JsonMessagePassingProfilingService,
                        ServiceProperty("JsonMessagePassingProfiling",ST_SubDomain),
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
  // creation du json writer
  if (!m_json_writer) {
    m_json_writer = new JSONWriter(JSONWriter::FormatFlags::None);
    m_json_writer->beginObject();
  }

  // Lorsqu'on lance le profiling on commence l'observation des evts du timeLoopMng
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
  // Arret des observations
  m_observer.detachAll();
  // On flush les infos
  if (m_json_writer)
    _dumpCurrentIterationInJSON();
  // On ferme l'objet
  m_json_writer->endObject();
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

  // Numero d'iteration (decale d'un car le compteur s'incremente avant la notification)
  JSONWriter::Object obj_it(*m_json_writer,
  		                      String::fromNumber(m_sub_domain->commonVariables().globalIteration() - 1));
  // Par point d'entree
  for (const auto &ep_mpstat : m_ep_mpstat_col) {
  	// Alias sur les stats de msg passing pour ce pt d'entree
  	const auto& stats_(ep_mpstat.second.stats());
  	// On ecrit le pt d'entree que s'il existe une stat de msg passing non nulle
  	if (std::any_of(stats_.cbegin(), stats_.cend(), [&](const Arccore::MessagePassing::OneStat& os)
	                                                  {return static_cast<bool>(os.nbMessage());})) {
  		// Nom du point d'entree
      JSONWriter::Object obj_ep(*m_json_writer, ep_mpstat.first);
      // Par stat msg passing
      for (const auto &mpstat : stats_) {
        // Stats message passing s'il y en a
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
  // On arrive dans le pt d'entree, on s'assure que les stats sont remises a zero
  m_sub_domain->parallelMng()->stat()->toArccoreStat()->resetCurrentStat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
_updateFromEndEntryPointEvt()
{
  // On recupere le nom du pt d'entree
  const String& ep_name(m_sub_domain->timeLoopMng()->currentEntryPoint()->name());
  auto pos(m_ep_mpstat_col.find(ep_name));
  // S'il n'y a pas encore de stats pour ce pt d'entree, on les crees
  Arccore::MessagePassing::StatData stat_data(m_sub_domain->parallelMng()->stat()->toArccoreStat()->stats());
  if (pos == m_ep_mpstat_col.end())
    m_ep_mpstat_col.emplace(ep_name, stat_data);
  else  // sinon, on merge les stats
    pos->second.mergeAllData(stat_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
_updateFromBeginIterationEvt()
{
  // On commence avec une structure vide, puisqu'elle est dumpee a chq fin d'iteration
  m_ep_mpstat_col.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JsonMessagePassingProfilingService::
_updateFromEndIterationEvt()
{
  // On dump tout ce qu'on a enregistre pour cette iteration
  if (m_json_writer)
    _dumpCurrentIterationInJSON();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
