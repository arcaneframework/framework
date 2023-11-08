// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueue.h                                                  (C) 2000-2023 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_RUNQUEUE_H
#define ARCANE_ACCELERATOR_CORE_RUNQUEUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/RunCommand.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief File d'exécution pour un accélérateur.
 *
 * Une file est attachée à une politique d'exécution et permet d'exécuter
 * des commandes (RunCommand) sur un accélérateur ou sur le CPU.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT RunQueue
{
  friend class RunCommand;
  friend class impl::RunCommandLaunchInfo;

 public:

  //! Permet de modifier l'asynchronisme de la file pendant la durée de vie de l'instance
  class ScopedAsync
  {
   public:

    explicit ScopedAsync(RunQueue* queue)
    : m_queue(queue)
    {
      // Rend la file asynchrone
      if (m_queue) {
        m_is_async = m_queue->isAsync();
        m_queue->setAsync(true);
      }
    }
    ~ScopedAsync() noexcept(false)
    {
      // Remet la file dans l'état d'origine lors de l'appel au constructeur
      if (m_queue)
        m_queue->setAsync(m_is_async);
    }

   private:

    RunQueue* m_queue = nullptr;
    bool m_is_async = false;
  };

 public:

  //! Créé une file associée à \a runner avec les paramètres par défaut
  explicit RunQueue(Runner& runner);
  //! Créé une file associée à \a runner avec les paramètres \a bi
  RunQueue(Runner& runner, const RunQueueBuildInfo& bi);
  ~RunQueue();

 public:

  RunQueue(const RunQueue&) = delete;
  RunQueue(RunQueue&&) = delete;
  RunQueue& operator=(const RunQueue&) = delete;
  RunQueue& operator=(RunQueue&&) = delete;

 public:

  //! Politique d'exécution de la file.
  eExecutionPolicy executionPolicy() const;
  /*!
   * \brief Positionne l'asynchronisme de l'instance.
   *
   * Si l'instance est asynchrone, il faut appeler explicitement barrier()
   * pour attendre la fin de l'exécution des commandes.
   */
  void setAsync(bool v) { m_is_async = v; }
  //! Indique si la file d'exécution est asynchrone.
  bool isAsync() const { return m_is_async; }
  //! Bloque tant que toutes les commandes associées à la file ne sont pas terminées.
  void barrier();

  //! Copie des informations entre deux zones mémoires
  void copyMemory(const MemoryCopyArgs& args);
  //! Effectue un préfetching de la mémoire
  void prefetchMemory(const MemoryPrefetchArgs& args);

  //! Enregistre l'état de l'instance dans \a event.
  void recordEvent(RunQueueEvent& event);
  //! Enregistre l'état de l'instance dans \a event.
  void recordEvent(Ref<RunQueueEvent>& event);
  //! Bloque l'exécution sur l'instance tant que les jobs enregistrés dans \a event ne sont pas terminés
  void waitEvent(RunQueueEvent& event);
  //! Bloque l'exécution sur l'instance tant que les jobs enregistrés dans \a event ne sont pas terminés
  void waitEvent(Ref<RunQueueEvent>& event);

 public:

  /*!
  * \brief Pointeur sur la structure interne dépendante de l'implémentation.
  *
  * Cette méthode est réservée à un usage avancée.
  * La file retournée ne doit pas être conservée au delà de la vie de l'instance.
  *
  * Avec CUDA, le pointeur retourné est un 'cudaStream_t*'. Avec HIP, il
  * s'agit d'un 'hipStream_t*'.
  */
  void* platformStream();

 private:

  impl::IRunnerRuntime* _internalRuntime() const;
  impl::IRunQueueStream* _internalStream() const;
  impl::RunCommandImpl* _getCommandImpl();

 private:

  impl::RunQueueImpl* m_p;
  bool m_is_async = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une commande associée à la file \a run_queue.
 */
inline RunCommand
makeCommand(RunQueue& run_queue)
{
  return RunCommand(run_queue);
}

/*!
 * \brief Créé une commande associée à la file \a run_queue.
 */
inline RunCommand
makeCommand(RunQueue* run_queue)
{
  ARCANE_CHECK_POINTER(run_queue);
  return RunCommand(*run_queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
