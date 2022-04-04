// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelTopology.cc                                         (C) 2000-2011 */
/*                                                                           */
/* Informations sur la topologie d'allocation des coeurs de calcul.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/IParallelMng.h"

#include "arcane/impl/ParallelTopology.h"

#include <map>
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelTopology::
ParallelTopology(IParallelMng* pm)
: m_parallel_mng(pm)
, m_machine_rank(-1)
, m_process_rank(-1)
, m_is_machine_master(false)
, m_is_process_master(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTopology::
initialize()
{
  // Test pour être sur que tout le monde appèle cette méthode d'initialisation.
  m_parallel_mng->barrier();
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
// Contient nom du noeud + pid pour connaitre les rangs maitres par processus.
class NamePid
{
 public:
  NamePid(ByteConstArrayView _name,Int64 _pid)
  : name(_name), pid(_pid){}
 public:
  ByteConstArrayView name;
  Int64 pid;
  bool operator<(const NamePid& b) const
  {
    int s = ::strcmp((const char*)name.data(),(const char*)b.name.data());
    if (s!=0)
      return s>0;
    return pid>b.pid;
    
  }

};
// Comparateur pour le nom du noeud.
class _Comparer
{
 public:
  bool operator()(ByteConstArrayView a,ByteConstArrayView b) const
  {
    return ::strcmp((const char*)a.data(),(const char*)b.data())>0;
  }
};
}

void ParallelTopology::
_init()
{
  IParallelMng* pm = m_parallel_mng;
  ITraceMng* tm = m_parallel_mng->traceMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();

  String host_name = platform::getHostName();
  Int64 pid = platform::getProcessId();
  // Tous les rangs qui ont le même nom \a host_name sont sur la même machine
  Integer len = host_name.utf8().size()+1;
  Integer max_len = pm->reduce(Parallel::ReduceMax,len);
  ByteUniqueArray all_names(max_len*nb_rank);
  ByteUniqueArray my_name;
  Int64UniqueArray all_pids(nb_rank);

  my_name.copy(host_name.utf8()); // copy shrink array size to arg size
  my_name.resize(max_len,'\0');   // add \0 to fill up to max_len

  pm->allGather(my_name,all_names);
  pm->allGather(Int64ConstArrayView(1,&pid),all_pids);

  m_machine_ranks.clear();
  m_process_ranks.clear();

  typedef std::map<ByteConstArrayView,Int32,_Comparer> MasterRankMap;
  typedef std::map<NamePid,Int32> MasterProcessRankMap;

  MasterRankMap machine_ranks_map;
  MasterProcessRankMap process_ranks_map;

  for( Integer irank=0; irank<nb_rank; ++irank ){
    ByteConstArrayView rank_name(max_len,&all_names[max_len*irank]);
    bool is_same_name = true;
    for( Integer j=0; j<max_len; ++j )
      if (rank_name[j]!=my_name[j]){
        is_same_name = false;
        break;
      }
    bool is_same_process = false;
    if (is_same_name && all_pids[irank]==pid)
      is_same_process = true;

    if (is_same_name)
      m_machine_ranks.add(irank);
    if (is_same_process)
      m_process_ranks.add(irank);
    {
      MasterRankMap::iterator i_master = machine_ranks_map.find(rank_name);
      if (i_master==machine_ranks_map.end()){
        // Comme le parcours des rangs est dans l'ordre des rangs,
        // le rang maitre est le premier rencontré.
        machine_ranks_map.insert(std::make_pair(rank_name,irank));
      }
    }

    {
      NamePid mp(rank_name,all_pids[irank]);
      MasterProcessRankMap::iterator i_master = process_ranks_map.find(mp);
      if (i_master==process_ranks_map.end()){
        process_ranks_map.insert(std::make_pair(mp,irank));
      }
    }

    tm->info(4) << "NAME RANK="<< irank << " n=" << (const char*)rank_name.data()
                << " same_rank=" << is_same_name
                << " same_process=" << is_same_process;
  }

  // Les rangs dans m_machine_ranks et m_process_ranks sont rangés
  // par ordre croissant. On considère que le maître est le premier
  // de la liste.
  if (m_machine_ranks[0]==my_rank)
    m_is_machine_master = true;
  if (m_process_ranks[0]==my_rank)
    m_is_process_master = true;

  m_master_machine_ranks.clear();
  for( MasterRankMap::const_iterator i(machine_ranks_map.begin()); i!=machine_ranks_map.end(); ++i )
    m_master_machine_ranks.add(i->second);
  std::sort(std::begin(m_master_machine_ranks),std::end(m_master_machine_ranks));
  for( Integer i=0, n=m_master_machine_ranks.size(); i<n; ++i ){
    if (m_master_machine_ranks[i]==m_machine_ranks[0]){
      m_machine_rank = i;
      break;
    }
  }

  m_master_process_ranks.clear();
  for( MasterProcessRankMap::const_iterator i(process_ranks_map.begin()); i!=process_ranks_map.end(); ++i )
    m_master_process_ranks.add(i->second);
  std::sort(std::begin(m_master_process_ranks),std::end(m_master_process_ranks));
  for( Integer i=0, n=m_master_process_ranks.size(); i<n; ++i ){
    if (m_master_process_ranks[i]==m_process_ranks[0]){
      m_process_rank = i;
      break;
    }
  }
  tm->info(4) << " MachineRank=" << m_machine_rank
              << " ProcessRank=" << m_process_rank;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* ParallelTopology::
parallelMng() const
{
  return m_parallel_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParallelTopology::
isMasterMachine() const
{
  return m_is_machine_master;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView ParallelTopology::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ParallelTopology::
machineRank() const
{
  return m_machine_rank;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParallelTopology::
isMasterProcess() const
{
  return m_is_process_master;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView ParallelTopology::
processRanks() const
{
  return m_process_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ParallelTopology::
processRank() const
{
  return m_process_rank;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView ParallelTopology::
masterMachineRanks() const
{
  return m_master_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView ParallelTopology::
masterProcessRanks() const
{
  return m_master_process_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
