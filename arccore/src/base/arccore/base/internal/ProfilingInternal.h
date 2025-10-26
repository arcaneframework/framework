// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ProfilingInternal.h                                         (C) 2000-2025 */
/*                                                                           */
/* Classes internes pour gérer le profilage.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_INTERNAL_PROFILINGINTERNAL_H
#define ARCCORE_BASE_INTERNAL_PROFILINGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Note: ce fichier n'est pas disponible pour les utilisateurs de Arcane.
// Il ne faut donc pas l'inclure dans un fichier d'en-tête public.

#include "arccore/base/String.h"
#include "arccore/base/FixedArray.h"

#include <map>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques d'exécution d'une boucle.
 */
struct ARCCORE_BASE_EXPORT ForLoopProfilingStat
{
 public:

  //! Ajoute les infos de l'exécution \a s
  void add(const ForLoopOneExecStat& s);

  Int64 nbCall() const { return m_nb_call; }
  Int64 nbChunk() const { return m_nb_chunk; }
  Int64 execTime() const { return m_exec_time; }

 private:

  Int64 m_nb_call = 0;
  Int64 m_nb_chunk = 0;
  Int64 m_exec_time = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_BASE_EXPORT ForLoopStatInfoListImpl
{
 public:

  void print(std::ostream& o);

 public:

  // TODO Utiliser un hash pour le map plutôt qu'une String pour accélérer les comparaisons
  std::map<String, ForLoopProfilingStat> m_stat_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques cumulées sur le nombre de boucles exécutées.
 */
class ARCCORE_BASE_EXPORT ForLoopCumulativeStat
{
 public:

  void merge(const ForLoopOneExecStat& s)
  {
    ++m_nb_loop_parallel_for;
    m_nb_chunk_parallel_for += s.nbChunk();
    m_total_time += s.execTime();
  }

 public:

  Int64 nbLoopParallelFor() const { return m_nb_loop_parallel_for.load(); }
  Int64 nbChunkParallelFor() const { return m_nb_chunk_parallel_for.load(); }
  Int64 totalTime() const { return m_total_time.load(); }

 private:

  std::atomic<Int64> m_nb_loop_parallel_for = 0;
  std::atomic<Int64> m_nb_chunk_parallel_for = 0;
  std::atomic<Int64> m_total_time = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques pour les accélérateurs.
 *
 * TODO: regarder comment rendre cela plus générique et permettre à
 * l'implémentation d'ajouter ses évènements
 */
class ARCCORE_BASE_EXPORT AcceleratorStatInfoList
{
 public:

  //! Informations sur les transferts mémoire entre CPU et GPU
  class MemoryTransferInfo
  {
   public:

    void merge(const MemoryTransferInfo& mem_info)
    {
      m_nb_byte += mem_info.m_nb_byte;
      m_nb_call += mem_info.m_nb_call;
    }

   public:

    Int64 m_nb_byte = 0;
    Int64 m_nb_call = 0;
  };

  //! Informations sur les défauts de page sur CPU ou GPU
  class MemoryPageFaultInfo
  {
   public:

    void merge(const MemoryPageFaultInfo& mem_info)
    {
      m_nb_fault += mem_info.m_nb_fault;
      m_nb_call += mem_info.m_nb_call;
    }

   public:

    Int64 m_nb_fault = 0;
    Int64 m_nb_call = 0;
  };

  enum class eMemoryTransferType
  {
    HostToDevice = 0,
    DeviceToHost = 1
  };
  enum class eMemoryPageFaultType
  {
    Gpu = 0,
    Cpu = 1
  };

 public:

  void addMemoryTransfer(eMemoryTransferType type, Int64 nb_byte)
  {
    MemoryTransferInfo mem_info{ nb_byte, 1 };
    m_managed_memory_transfer_list[(int)type].merge(mem_info);
  }
  const MemoryTransferInfo& memoryTransfer(eMemoryTransferType type) const
  {
    return m_managed_memory_transfer_list[(int)type];
  }
  void addMemoryPageFault(eMemoryPageFaultType type, Int64 nb_byte)
  {
    MemoryPageFaultInfo mem_info{ nb_byte, 1 };
    m_managed_memory_page_fault_list[(int)type].merge(mem_info);
  }
  const MemoryPageFaultInfo& memoryPageFault(eMemoryPageFaultType type) const
  {
    return m_managed_memory_page_fault_list[(int)type];
  }

 public:

  void print(std::ostream& ostr) const;

 private:

  // Doit avoir le même nombre d'éléments que 'eMemoryTransfertType'
  FixedArray<MemoryTransferInfo, 2> m_managed_memory_transfer_list;

  // Doit avoir le même nombre d'éléments que 'eMemoryPageFaultType'
  FixedArray<MemoryPageFaultInfo, 2> m_managed_memory_page_fault_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
