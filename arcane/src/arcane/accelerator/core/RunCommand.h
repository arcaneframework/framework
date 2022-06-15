// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommand.h                                                (C) 2000-2022 */
/*                                                                           */
/* Gestion d'une commande sur accélérateur.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_RUNCOMMAND_H
#define ARCANE_ACCELERATOR_CORE_RUNCOMMAND_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion d'une commande sur accélérateur.
 *
 * Une commande est associée à une file d'exécution (RunQueue) et sa durée
 * de vie ne doit pas excéder celle de cette dernière.
 *
 * \warning API en cours de définition.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT RunCommand
{
  friend impl::IReduceMemoryImpl* impl::internalGetOrCreateReduceMemoryImpl(RunCommand* command);
  friend impl::RunCommandLaunchInfo;

 public:

  RunCommand(RunQueue& run_queue);
  RunCommand(const RunCommand&) = delete;
  RunCommand& operator=(const RunCommand&) = delete;
  ~RunCommand();

 public:

  RunCommand& addTraceInfo(const TraceInfo& ti);
  RunCommand& addKernelName(const String& v);
  const TraceInfo& traceInfo() const;
  const String& kernelName() const;
  friend ARCANE_ACCELERATOR_CORE_EXPORT RunCommand& operator<<(RunCommand& command,const TraceInfo& trace_info);

 private:

  // Uniquement pour RunCommandLaunchInfo
  void _resetInfos();

 public:

  //! \internal
  RunQueue& _internalQueue() { return m_run_queue; }
  static impl::RunCommandImpl* _internalCreateImpl(impl::RunQueueImpl* queue);
  static void _internalDestroyImpl(impl::RunCommandImpl* p);

 private:

  RunQueue& m_run_queue;
  impl::RunCommandImpl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
