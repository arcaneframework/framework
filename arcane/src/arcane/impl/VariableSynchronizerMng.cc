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
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/internal/MemoryBuffer.h"

#include "arcane/core/IVariableMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableSynchronizerEventArgs.h"
#include "arcane/core/IVariable.h"

#include <map>
#include <stack>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques de synchronisation.
 *
 * Lorsque la comparaison avant/après synchronisation est active, chaque rang
 * sait pour sa partie si les valeurs comparées sont les mêmes ou pas.
 *
 * Cependant, il faut faire une réduction sur l'ensemble des rangs pour avoir
 * une vision globale de la comparaison (car il suffit d'un rang pour lequel
 * la comparaison est différente pour qu'on considère que la comparaison est
 * différente).
 *
 * Comme il est trop coûteux de faire la réduction pour chaque synchronisation,
 * on maintient une liste des comparaisons et on traite cette liste lorsqu'elle
 * atteint une certaine taille ou si c'est demandé explicitement.
 *
 */
class VariableSynchronizerStats
: public TraceAccessor
{
 public:

  // On utilise un ReduceMin pour la valeur de comparaison.
  // Pour qu'on considère comme identique, il faut que tout les rangs
  // soient identiques. Il suffit d'un rang 'Unknown' pour considérer
  // que c'est 'Unknown'. Il faut donc que 'Unknown' soit la valeur la
  // plus faible et 'Same' la plus élevée.
  static constexpr unsigned char LOCAL_UNKNOWN = 0;
  static constexpr unsigned char LOCAL_DIFF = 1;
  static constexpr unsigned char LOCAL_SAME = 2;

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
    if (m_is_event_registered)
      ARCANE_FATAL("instance is already initialized.");
    auto handler = [&](const VariableSynchronizerEventArgs& args) {
      _handleEvent(args);
    };
    vsm->onSynchronized().attach(m_observer_pool, handler);
    m_is_event_registered = true;
  }

  void flushPendingStats(IParallelMng* pm);

  void dumpStats(std::ostream& ostr)
  {
    std::streamsize old_precision = ostr.precision(20);
    ostr << "Synchronization Stats\n";
    ostr << Trace::Width(50) << "Variable name"
         << Trace::Width(8) << "Count"
         << Trace::Width(8) << "NbSame"
         << Trace::Width(8) << "NbDiff"
         << "\n";
    for (const auto& p : m_stats) {
      ostr << Trace::Width(50) << p.first
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
  UniqueArray<String> m_pending_variable_name_list;
  UniqueArray<unsigned char> m_pending_compare_status_list;

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
        m_pending_variable_name_list.add(var->fullName());
        VariableSynchronizerEventArgs::CompareStatus s = compare_status_list[index];
        unsigned char rs = LOCAL_UNKNOWN; // Compare == Unknown;
        if (s == VariableSynchronizerEventArgs::CompareStatus::Same)
          rs = LOCAL_SAME;
        else if (s == VariableSynchronizerEventArgs::CompareStatus::Different)
          rs = LOCAL_DIFF;
        m_pending_compare_status_list.add(rs);
        ++index;
      }
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerStats::
flushPendingStats(IParallelMng* pm)
{
  Int32 nb_pending = m_pending_variable_name_list.size();
  Int32 total_nb_pending = pm->reduce(Parallel::ReduceMax, nb_pending);
  if (total_nb_pending != nb_pending)
    ARCANE_FATAL("Bad number of pending stats local={0} global={1}", nb_pending, total_nb_pending);
  pm->reduce(Parallel::ReduceMin, m_pending_compare_status_list);
  for (Int32 i = 0; i < total_nb_pending; ++i) {
    unsigned char rs = m_pending_compare_status_list[i];
    auto& v = m_stats[m_pending_variable_name_list[i]];
    if (rs == LOCAL_SAME)
      ++v.m_nb_same;
    else if (rs == LOCAL_DIFF)
      ++v.m_nb_different;
    ++v.m_count;
  }
  m_pending_variable_name_list.clear();
  m_pending_compare_status_list.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerMng::
VariableSynchronizerMng(IVariableMng* vm)
: TraceAccessor(vm->traceMng())
, m_variable_mng(vm)
, m_parallel_mng(vm->parallelMng())
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
  m_internal_api.dumpStats(ostr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::
flushPendingStats()
{
  if (m_is_compare_synchronize)
    m_stats->flushPendingStats(m_parallel_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère un pool de buffer associé à un allocateur.
 */
class VariableSynchronizerMng::InternalApi::BufferList
{
 public:

  using MemoryBufferMap = std::map<MemoryBuffer*, Ref<MemoryBuffer>>;
  using MapList = std::map<IMemoryAllocator*, MemoryBufferMap>;

  using FreeList = std::map<IMemoryAllocator*, std::stack<Ref<MemoryBuffer>>>;

 public:

  //! Liste par allocateur des buffers en cours d'utilisation
  MapList m_used_map;

  //! Liste par allocateur des buffers libres
  FreeList m_free_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerMng::InternalApi::
InternalApi(VariableSynchronizerMng* vms)
: TraceAccessor(vms->traceMng())
, m_synchronizer_mng(vms)
, m_buffer_list(new BufferList())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerMng::InternalApi::
~InternalApi()
{
  delete m_buffer_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Créé ou récupère un buffer.
 *
 * Il est possible de créer des buffers avec un allocateur nul. Dans ce
 * cas ce sera l'allocateur par défaut qui sera utilisé et donc pour
 * un MemoryBuffer donné, on n'aura pas forcément new_buffer.allocator()==allocator.
 * Il ne faut donc toujours utiliser \a allocator.
 */
Ref<MemoryBuffer> VariableSynchronizerMng::InternalApi::
createSynchronizeBuffer(IMemoryAllocator* allocator)
{
  auto& free_map = m_buffer_list->m_free_map;
  auto x = free_map.find(allocator);
  Ref<MemoryBuffer> new_buffer;
  // Regarde si un buffer est disponible dans \a free_map.
  if (x == free_map.end()) {
    // Aucune buffer associé à cet allocator, on en créé un
    new_buffer = MemoryBuffer::create(allocator);
  }
  else {
    auto& buffer_stack = x->second;
    // Si la pile est vide, on créé un buffer. Sinon on prend le premier
    // de la pile.
    if (buffer_stack.empty()) {
      new_buffer = MemoryBuffer::create(allocator);
    }
    else {
      new_buffer = buffer_stack.top();
      buffer_stack.pop();
    }
  }

  // Enregistre l'instance dans la liste utilisée
  m_buffer_list->m_used_map[allocator].insert(std::make_pair(new_buffer.get(), new_buffer));
  return new_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::
releaseSynchronizeBuffer(IMemoryAllocator* allocator, MemoryBuffer* v)
{
  auto& main_map = m_buffer_list->m_used_map;
  auto x = main_map.find(allocator);
  if (x == main_map.end())
    ARCANE_FATAL("Invalid allocator '{0}'", allocator);

  auto& sub_map = x->second;
  auto x2 = sub_map.find(v);
  if (x2 == sub_map.end())
    ARCANE_FATAL("Invalid buffer '{0}'", v);

  Ref<MemoryBuffer> ref_memory = x2->second;

  sub_map.erase(x2);

  m_buffer_list->m_free_map[allocator].push(ref_memory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::
dumpStats(std::ostream& ostr) const
{
  //! Liste par allocateur des buffers en cours d'utilisation
  for (const auto& x : m_buffer_list->m_used_map)
    ostr << "SynchronizeBuffer: nb_used_map = " << x.second.size() << "\n";

  //! Liste par allocateur des buffers libres
  for (const auto& x : m_buffer_list->m_free_map)
    ostr << "SynchronizeBuffer: nb_free_map = " << x.second.size() << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
