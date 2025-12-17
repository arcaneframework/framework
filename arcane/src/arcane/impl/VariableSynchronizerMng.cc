// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerMng.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire des synchroniseurs de variables.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableSynchronizerMng.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/internal/MemoryBuffer.h"

#include "arcane/core/IVariableMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableSynchronizerEventArgs.h"
#include "arcane/core/IVariable.h"

#include <map>
#include <mutex>
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

    void add(const StatInfo& x)
    {
      m_count += x.m_count;
      m_nb_same += x.m_nb_same;
      m_nb_different += x.m_nb_different;
      m_nb_unknown += x.m_nb_unknown;
    }

   public:

    Int32 m_count = 0;
    Int32 m_nb_same = 0;
    Int32 m_nb_different = 0;
    Int32 m_nb_unknown = 0;
  };

 public:

  explicit VariableSynchronizerStats(VariableSynchronizerMng* vsm)
  : TraceAccessor(vsm->traceMng())
  , m_variable_synchronizer_mng(vsm)
  {}

 public:

  void init()
  {
    if (m_is_event_registered)
      ARCANE_FATAL("instance is already initialized.");
    auto handler = [&](const VariableSynchronizerEventArgs& args) {
      _handleEvent(args);
    };
    m_variable_synchronizer_mng->onSynchronized().attach(m_observer_pool, handler);
    m_is_event_registered = true;
  }

  void flushPendingStats(IParallelMng* pm);

  Int32 dumpStats(std::ostream& ostr)
  {
    std::streamsize old_precision = ostr.precision(20);
    ostr << "Synchronization Stats\n";
    ostr << Trace::Width(8) << "Total"
         << Trace::Width(8) << "  Nb "
         << Trace::Width(8) << "  Nb "
         << Trace::Width(8) << " Nb  "
         << "   Variable name"
         << "\n";
    ostr << Trace::Width(8) << "Count"
         << Trace::Width(8) << "Same"
         << Trace::Width(8) << "Diff"
         << Trace::Width(8) << "Unknown"
         << "\n";
    StatInfo total_stat;
    for (const auto& p : m_stats) {
      total_stat.add(p.second);
      ostr << " " << Trace::Width(7) << p.second.m_count
           << " " << Trace::Width(7) << p.second.m_nb_same
           << " " << Trace::Width(7) << p.second.m_nb_different
           << " " << Trace::Width(7) << p.second.m_nb_unknown
           << "   " << p.first
           << "\n";
    }
    ostr << "\n";
    ostr << " " << Trace::Width(7) << total_stat.m_count
         << " " << Trace::Width(7) << total_stat.m_nb_same
         << " " << Trace::Width(7) << total_stat.m_nb_different
         << " " << Trace::Width(7) << total_stat.m_nb_unknown
         << "   "
         << "TOTAL"
         << "\n\n";
    ostr.precision(old_precision);
    return total_stat.m_count;
  }

 private:

  VariableSynchronizerMng* m_variable_synchronizer_mng = nullptr;
  EventObserverPool m_observer_pool;
  std::map<String, StatInfo> m_stats;
  bool m_is_event_registered = false;
  UniqueArray<String> m_pending_variable_name_list;
  UniqueArray<unsigned char> m_pending_compare_status_list;

 private:

  void _handleEvent(const VariableSynchronizerEventArgs& args);
};

void VariableSynchronizerStats::
_handleEvent(const VariableSynchronizerEventArgs& args)
{
  // On ne traite que les évènements de fin de synchronisation
  if (args.state() != VariableSynchronizerEventArgs::State::EndSynchronize)
    return;
  if (!m_variable_synchronizer_mng->isDoingStats())
    return;
  Int32 level = m_variable_synchronizer_mng->synchronizationCompareLevel();
  IParallelMng* pm = m_variable_synchronizer_mng->parallelMng();
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
      if (level >= 2) {
        // On fait la réduction ici car on veut savoir immédiatement s'il y a une
        // différence.
        unsigned char global_rs = pm->reduce(Parallel::ReduceMax, rs);
        if (global_rs == LOCAL_SAME) {
          info() << "Synchronize: same values for variable name=" << var->fullName();
          if (level >= 3)
            info() << "Stack=" << platform::getStackTrace();
        }
      }
    }
  }
}

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
    else
      ++v.m_nb_unknown;
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
, m_stats(new VariableSynchronizerStats(this))
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_AUTO_COMPARE_SYNCHRONIZE", true)) {
    m_synchronize_compare_level = v.value();
    // Si on active la comparaison, on active aussi les statistiques
    m_is_doing_stats = m_synchronize_compare_level > 0;
  }
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_SYNCHRONIZE_STATS", true))
    m_is_doing_stats = (v.value() != 0);
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
  m_stats->init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::
dumpStats(std::ostream& ostr) const
{
  if (!m_parallel_mng->isParallel())
    return;
  {
    OStringStream ostr2;
    Int32 count = m_stats->dumpStats(ostr2());
    if (count > 0)
      ostr << ostr2.str();
  }
  m_internal_api.dumpStats(ostr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::
flushPendingStats()
{
  if (isDoingStats())
    m_stats->flushPendingStats(m_parallel_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère un pool de buffer associé à un allocateur.
 *
 * Les méthodes de cette classe sont thread-safe.
 */
class VariableSynchronizerMng::InternalApi::BufferList
{
 public:

  using MemoryBufferMap = std::map<MemoryBuffer*, Ref<MemoryBuffer>>;
  using MapList = std::map<IMemoryAllocator*, MemoryBufferMap>;

  using FreeList = std::map<IMemoryAllocator*, std::stack<Ref<MemoryBuffer>>>;

 public:

  Ref<MemoryBuffer> createSynchronizeBuffer(IMemoryAllocator* allocator);
  void releaseSynchronizeBuffer(IMemoryAllocator* allocator, MemoryBuffer* v);
  void dumpStats(std::ostream& ostr) const;

 private:

  //! Liste par allocateur des buffers en cours d'utilisation
  MapList m_used_map;

  //! Liste par allocateur des buffers libres
  FreeList m_free_map;

  //! Mutex pour protéger la création/récupération des buffers
  mutable std::mutex m_mutex;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Créé ou récupère un buffer.
 *
 * Il est possible de créer des buffers avec un allocateur nul. Dans ce
 * cas, ce sera l'allocateur par défaut qui sera utilisé et donc pour
 * un MemoryBuffer donné, on n'aura pas forcément new_buffer.allocator()==allocator.
 * Il faut donc toujours utiliser \a allocator.
 */
Ref<MemoryBuffer> VariableSynchronizerMng::InternalApi::BufferList::
createSynchronizeBuffer(IMemoryAllocator* allocator)
{
  std::scoped_lock lock(m_mutex);

  auto& free_map = m_free_map;
  auto x = free_map.find(allocator);
  Ref<MemoryBuffer> new_buffer;
  // Regarde si un buffer est disponible dans \a free_map.
  if (x == free_map.end()) {
    // Aucun buffer associé à cet allocator, on en crée un
    new_buffer = MemoryBuffer::create(allocator);
  }
  else {
    auto& buffer_stack = x->second;
    // Si la pile est vide, on crée un buffer. Sinon, on prend le premier
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
  m_used_map[allocator].insert(std::make_pair(new_buffer.get(), new_buffer));
  return new_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::BufferList::
releaseSynchronizeBuffer(IMemoryAllocator* allocator, MemoryBuffer* v)
{
  std::scoped_lock lock(m_mutex);

  auto& main_map = m_used_map;
  auto x = main_map.find(allocator);
  if (x == main_map.end())
    ARCANE_FATAL("Invalid allocator '{0}'", allocator);

  auto& sub_map = x->second;
  auto x2 = sub_map.find(v);
  if (x2 == sub_map.end())
    ARCANE_FATAL("Invalid buffer '{0}'", v);

  Ref<MemoryBuffer> ref_memory = x2->second;

  sub_map.erase(x2);

  m_free_map[allocator].push(ref_memory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::BufferList::
dumpStats(std::ostream& ostr) const
{
  std::scoped_lock lock(m_mutex);

  //! Liste par allocateur des buffers en cours d'utilisation
  for (const auto& x : m_used_map)
    ostr << "SynchronizeBuffer: nb_used_map = " << x.second.size() << "\n";

  //! Liste par allocateur des buffers libres
  for (const auto& x : m_free_map)
    ostr << "SynchronizeBuffer: nb_free_map = " << x.second.size() << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
  // Le destructeur ne peut pas être supprimé car 'm_buffer_list' n'est pas
  // connu lors de la définition de la classe.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<MemoryBuffer> VariableSynchronizerMng::InternalApi::
createSynchronizeBuffer(IMemoryAllocator* allocator)
{
  return m_buffer_list->createSynchronizeBuffer(allocator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::
releaseSynchronizeBuffer(IMemoryAllocator* allocator, MemoryBuffer* v)
{
  m_buffer_list->releaseSynchronizeBuffer(allocator, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMng::InternalApi::
dumpStats(std::ostream& ostr) const
{
  m_buffer_list->dumpStats(ostr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
