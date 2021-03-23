// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SequentialParallelMng.h                                     (C) 2000-2020 */
/*                                                                           */
/* Gestion du parallélisme dans le cas sequentiel.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_SEQUENTIALPARALLELMNG_H
#define ARCANE_IMPL_SEQUENTIALPARALLELMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeMng;
class IThreadMng;
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour construire un SequentialParallelMng.
 *
 * Il est indispensable d'appeler setTraceMng() et setThreadMng(). Les
 * autres méthodes sont obsolètes.
 */
struct SequentialParallelMngBuildInfo
{
 public:
  SequentialParallelMngBuildInfo()
  : m_trace_mng(nullptr), m_timer_mng(nullptr), m_thread_mng(nullptr)
  {
  }
  SequentialParallelMngBuildInfo(ITimerMng* timer_mng,IParallelMng* world_pm)
  : m_trace_mng(nullptr), m_timer_mng(timer_mng),
    m_thread_mng(nullptr), m_world_parallel_mng(world_pm)
  {
  }
 public:
  ITraceMng* traceMng() const { return m_trace_mng; }
  void setTraceMng(ITraceMng* tm)
  {
    m_trace_mng = tm;
  }

  Parallel::Communicator communicator() const { return m_mpi_comm; }
  void setCommunicator(Parallel::Communicator v) { m_mpi_comm = v; }

  IThreadMng* threadMng() const { return m_thread_mng; }
  void setThreadMng(IThreadMng* tm) { m_thread_mng = tm; }
 private:
  ITraceMng* m_trace_mng;
 public:
  ITimerMng* m_timer_mng;
 private:
  IThreadMng* m_thread_mng;
 public:
  IParallelMng* m_world_parallel_mng = nullptr;
 private:
  Parallel::Communicator m_mpi_comm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++"
ARCCORE_DEPRECATED_2020("Use arcaneCreateSequentialParallelMngRef() instead")
ARCANE_IMPL_EXPORT IParallelMng*
arcaneCreateSequentialParallelMng(const SequentialParallelMngBuildInfo& bi);
extern "C++" ARCANE_IMPL_EXPORT Ref<IParallelMng>
arcaneCreateSequentialParallelMngRef(const SequentialParallelMngBuildInfo& bi);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

