// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommand.h                                                (C) 2000-2025 */
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
namespace impl
{
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion d'une commande sur accélérateur.
 *
 * Une commande est associée à une file d'exécution (RunQueue) et sa durée
 * de vie ne doit pas excéder celle de cette dernière.
 *
 * Une commande est une opération qui sera exécutée sur l'accélérateur
 * associé à l'instance de RunQueue utilisé lors de l'appel à makeCommand().
 * Sur un GPU, cela correspond à un noyau (kernel).
 *
 * Pour plus d'informations, se reporter à la rubrique
 * \ref arcanedoc_parallel_accelerator_runcommand.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT RunCommand
{
  friend impl::IReduceMemoryImpl* impl::internalGetOrCreateReduceMemoryImpl(RunCommand* command);
  friend impl::RunCommandLaunchInfo;
  friend impl::RunQueueImpl;
  friend class ViewBuildInfo;

  friend RunCommand makeCommand(const RunQueue& run_queue);
  friend RunCommand makeCommand(const RunQueue* run_queue);

 public:

  ~RunCommand();

 protected:

  explicit RunCommand(const RunQueue& run_queue);

 public:

  RunCommand(RunCommand&& command) = delete;
  RunCommand(const RunCommand&) = delete;
  RunCommand& operator=(const RunCommand&) = delete;
  RunCommand& operator=(RunCommand&&) = delete;

 public:

  /*!
   * \brief Positionne le informations de trace.
   *
   * Ces informations sont utilisées pour les traces ou pour le débug.
   * Les macros RUNCOMMAND_LOOP ou RUNCOMMAND_ENUMERATE appellent
   * automatiquement cette méthode.
   */
  RunCommand& addTraceInfo(const TraceInfo& ti);

  /*!
   * \brief Positionne le nom du noyau.
   *
   * Ce nom est utilisé pour les traces ou pour le débug.
   */
  RunCommand& addKernelName(const String& v);

  /*!
   * \brief Positionne le nombre de thread par bloc pour les accélérateurs.
   *
   * Si la valeur \a v est nulle, le choix par défaut est utilisé.
   * Si la valeur \a v est positive, sa valeur minimale valide dépend
   * de l'accélérateur. En général c'est au moins 32.
   */
  RunCommand& addNbThreadPerBlock(Int32 v);

  //! Informations pour les traces
  const TraceInfo& traceInfo() const;

  //! Nom du noyau
  const String& kernelName() const;

  /*
   * \brief Nombre de threads par bloc ou 0 pour la valeur par défaut.
   *
   * Cette valeur est utilisée uniquement si on s'exécute sur accélérateur.
   */
  Int32 nbThreadPerBlock() const;

  //! Positionne la configuration des boucles multi-thread
  void setParallelLoopOptions(const ParallelLoopOptions& opt);

  //! Configuration des boucles multi-thread
  const ParallelLoopOptions& parallelLoopOptions() const;

  //! Affichage des informations de la commande
  friend ARCANE_ACCELERATOR_CORE_EXPORT RunCommand&
  operator<<(RunCommand& command, const TraceInfo& trace_info);

 private:

  // Pour RunCommandLaunchInfo
  void _internalNotifyBeginLaunchKernel();
  void _internalNotifyEndLaunchKernel();
  void _internalNotifyBeginLaunchKernelSyclEvent(void* sycl_event_ptr);
  ForLoopOneExecStat* _internalCommandExecStat();

 private:

  //! \internal
  impl::RunQueueImpl* _internalQueueImpl() const;
  impl::NativeStream _internalNativeStream() const;
  static impl::RunCommandImpl* _internalCreateImpl(impl::RunQueueImpl* queue);
  static void _internalDestroyImpl(impl::RunCommandImpl* p);

 private:

  void _allocateReduceMemory(Int32 nb_grid);

 private:

  impl::RunCommandImpl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
