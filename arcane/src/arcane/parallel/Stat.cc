// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Stat.cc                                                     (C) 2000-2017 */
/*                                                                           */
/* Statistiques sur le parallélisme.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/Convert.h"

#include "arcane/parallel/IStat.h"
#include "arccore/message_passing/Stat.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_BEGIN_NAMESPACE_PARALLEL

using Arccore::MessagePassing::OneStat;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques sur le parallélisme.
 */
class Stat
: public Arccore::MessagePassing::Stat
, public IStat
{
 public:
  typedef Arccore::MessagePassing::Stat Base;

  Stat();
  //! Libère les ressources.
  virtual ~Stat();

  Arccore::MessagePassing::IStat* toArccoreStat() override { return this; }

  void add(const String& name,double elapsed_time,Int64 msg_size) override;
  void print(ITraceMng* msg) override;
  void enable(bool is_enabled) override { Base::enable(is_enabled); }
  void dumpJSON(JSONWriter& writer) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT IStat*
createDefaultStat()
{
  return new Stat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT void
dumpJSON(JSONWriter& writer, const Arccore::MessagePassing::OneStat& os, bool cumulative_stat)
{
    Arcane::JSONWriter::Object o(writer, os.name());
    writer.write("Count", cumulative_stat?os.cumulativeNbMessage():os.nbMessage());
    writer.write("MessageSize", cumulative_stat?os.cumulativeTotalSize():os.totalSize());
    writer.write("TotalTime", cumulative_stat?os.cumulativeTotalTime():os.totalTime());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Stat::
Stat()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Stat::
~Stat()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
add(const String& name,double elapsed_time,Int64 msg_size)
{
  Arccore::MessagePassing::Stat::add(name,elapsed_time,msg_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
print(ITraceMng* msg)
{
  for( auto i : stats() ){
    OneStat* os = i.second;
    Real total_time = os->cumulativeTotalTime();
    Int64 div_time = static_cast<Int64>(total_time*1000.0);
    Int64 nb_message = os->cumulativeNbMessage();
    Int64 total_size = os->cumulativeTotalSize();
    const String& name = os->name();
    if (div_time==0)
      div_time = 1;
    if (nb_message>0){
      Int64 average_time = Convert::toInt64(total_time/(Real)nb_message);
      msg->info() << " MPIStat " << name << "     :" << nb_message << " messages";
      msg->info() << " MPIStat " << name << "     :" << total_size << " bytes ("
                  << total_size / div_time << " Kb/s) (average size "
                  << total_size / nb_message << " bytes)";
      msg->info() << " MPIStat " << name << " Time: " << total_time << " seconds"
                  << " (avg=" << average_time << ")";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Stat::
dumpJSON(JSONWriter& writer)
{
  writer.writeKey("Stats");
  writer.beginArray();
  for (const auto& stat : stats())
    Parallel::dumpJSON(writer, *(stat.second));  // cumulative stats dump
  writer.endArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE_PARALLEL
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
