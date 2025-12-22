// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueue.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Gestion d'une file d'exécution sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNQUEUE_H
#define ARCCORE_COMMON_ACCELERATOR_RUNQUEUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/AutoRef2.h"

#include "arccore/common/accelerator/RunCommand.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief File d'exécution pour un accélérateur.
 *
 * Cette classe utilise une sémantique par référence. La file d'exécution est
 * détruite lorsque la dernière référence dessus est détruite.
 *
 * Une file est attachée à une instance de Runner et permet d'exécuter
 * des commandes (RunCommand) sur un accélérateur ou sur le CPU. La méthode
 * executionPolicy() permet de savoir où s'exécutera les commandes issues
 * de la file.
 *
 * Les instances de cette classe sont créées par l'appel à makeQueue(Runner).
 * On peut ensuite créer des noyaux de calcul (RunCommand) via l'appel
 * à makeCommand().
 *
 * Le constructeur par défaut construit Une file nulle qui ne peut pas être
 * utilisée pour lancer des commandes. Les seules opérations autorisées sur
 * la file nulle sont isNull(), executionPolicy(), isAcceleratorPolicy(),
 * barrier(), allocationOptions() et memoryResource().
 *
 * Les méthodes de cette classe ne sont pas thread-safe pour une même instance.
 */
class ARCCORE_COMMON_EXPORT RunQueue
{
  friend RunCommand;
  friend ProfileRegion;
  friend Runner;
  friend ViewBuildInfo;
  friend class Impl::RunCommandLaunchInfo;
  friend RunCommand makeCommand(const RunQueue& run_queue);
  friend RunCommand makeCommand(const RunQueue* run_queue);
  // Pour _internalNativeStream()
  friend class Impl::CudaUtils;
  friend class Impl::HipUtils;
  friend class Impl::SyclUtils;

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

  //! Créé une file nulle.
  RunQueue();
  ~RunQueue();

 public:

  //! Créé une file associée à \a runner avec les paramètres par défaut
  ARCCORE_DEPRECATED_REASON("Y2024: Use makeQueue(runner) instead")
  explicit RunQueue(const Runner& runner);
  //! Créé une file associée à \a runner avec les paramètres \a bi
  ARCCORE_DEPRECATED_REASON("Y2024: Use makeQueue(runner,bi) instead")
  RunQueue(const Runner& runner, const RunQueueBuildInfo& bi);

 public:

  RunQueue(const RunQueue&);
  RunQueue& operator=(const RunQueue&);
  RunQueue(RunQueue&&) noexcept;
  RunQueue& operator=(RunQueue&&) noexcept;

 public:

  //! Indique si la RunQueue est nulle
  bool isNull() const { return !m_p; }

  //! Politique d'exécution de la file.
  eExecutionPolicy executionPolicy() const;
  //! Indique si l'instance est associée à un accélérateur
  bool isAcceleratorPolicy() const;

  /*!
   * \brief Positionne l'asynchronisme de l'instance.
   *
   * Si l'instance est asynchrone, les différentes commandes
   * associées ne sont pas bloquantes et il faut appeler explicitement barrier()
   * pour attendre la fin de l'exécution des commandes.
   *
   * \pre !isNull()
   */
  void setAsync(bool v);
  //! Indique si la file d'exécution est asynchrone.
  bool isAsync() const;

  /*!
   * \brief Positionne l'asynchronisme de l'instance.
   *
   * Retourne l'instance.
   *
   * \pre !isNull()
   * \sa setAsync().
   */
  const RunQueue& addAsync(bool is_async) const;

  //! Bloque tant que toutes les commandes associées à la file ne sont pas terminées.
  void barrier() const;

  //! Copie des informations entre deux zones mémoires
  void copyMemory(const MemoryCopyArgs& args) const;
  //! Effectue un préfetching de la mémoire
  void prefetchMemory(const MemoryPrefetchArgs& args) const;

  /*!
   * \name Gestion des évènements
   * \pre !isNull()
   */
  //!@{
  //! Enregistre l'état de l'instance dans \a event.
  void recordEvent(RunQueueEvent& event);
  //! Enregistre l'état de l'instance dans \a event.
  void recordEvent(Ref<RunQueueEvent>& event);
  //! Bloque l'exécution sur l'instance tant que les jobs enregistrés dans \a event ne sont pas terminés
  void waitEvent(RunQueueEvent& event);
  //! Bloque l'exécution sur l'instance tant que les jobs enregistrés dans \a event ne sont pas terminés
  void waitEvent(Ref<RunQueueEvent>& event);
  //!@}

  //! \name Gestion mémoire
  //!@{
  /*!
   * \brief Options d'allocation associée à cette file.
   *
   * Il est possible de changer la ressource mémoire et donc l'allocateur utilisé
   * via setMemoryRessource().
   */
  MemoryAllocationOptions allocationOptions() const;

  /*!
   * \brief Positionne la ressource mémoire utilisée pour les allocations avec cette instance.
   *
   * La valeur par défaut est eMemoryRessource::UnifiedMemory
   * si isAcceleratorPolicy()==true et eMemoryRessource::Host sinon.
   *
   * \sa memoryResource()
   * \sa allocationOptions()
   *
   * \pre !isNull()
   */
  void setMemoryRessource(eMemoryResource mem);

  //! Ressource mémoire utilisée pour les allocations avec cette instance.
  eMemoryResource memoryRessource() const;
  //! Ressource mémoire utilisée pour les allocations avec cette instance.
  eMemoryResource memoryResource() const;
  //!@}

 public:

  /*!
   * \brief Indique si on autorise la création de RunCommand pour cette instance
   * depuis plusieurs threads.
   *
   * Cela nécessite d'utiliser un verrou (comme std::mutex) et peut dégrader les
   * performances. Le défaut est \a false.
   *
   * Cette méthode n'est pas supportée pour les files qui sont associées
   * à des accélérateurs (isAcceleratorPolicy()==true)
   */
  void setConcurrentCommandCreation(bool v);
  //! Indique si la création concurrente de plusieurs RunCommand est autorisée
  bool isConcurrentCommandCreation() const;

 public:

  /*!
   * \brief Pointeur sur la structure interne dépendante de l'implémentation.
   *
   * Cette méthode est réservée à un usage avancée.
   * La file retournée ne doit pas être conservée au delà de la vie de l'instance.
   *
   * Avec CUDA, le pointeur retourné est un 'cudaStream_t*'. Avec HIP, il
   * s'agit d'un 'hipStream_t*'.
   *
   * \deprecated Utiliser toCudaNativeStream(), toHipNativeStream()
   * ou toSyclNativeStream() à la place
   */
  ARCCORE_DEPRECATED_REASON("Y2024: Use toCudaNativeStream(), toHipNativeStream() or toSyclNativeStream() instead")
  void* platformStream() const;

 public:

  friend bool operator==(const RunQueue& q1, const RunQueue& q2)
  {
    return q1.m_p.get() == q2.m_p.get();
  }
  friend bool operator!=(const RunQueue& q1, const RunQueue& q2)
  {
    return q1.m_p.get() != q2.m_p.get();
  }

 public:

  impl::RunQueueImpl* _internalImpl() const;

 private:

  // Les méthodes de création sont réservée à Runner.
  // On ajoute un argument supplémentaire non utilisé pour ne pas utiliser
  // le constructeur obsolète.
  RunQueue(const Runner& runner, bool);
  //! Créé une file associée à \a runner avec les paramètres \a bi
  RunQueue(const Runner& runner, const RunQueueBuildInfo& bi, bool);
  explicit RunQueue(impl::RunQueueImpl* p);

 private:

  impl::IRunnerRuntime* _internalRuntime() const;
  impl::IRunQueueStream* _internalStream() const;
  impl::RunCommandImpl* _getCommandImpl() const;
  Impl::NativeStream _internalNativeStream() const;
  void _checkNotNull() const;

  // Pour VariableViewBase
  friend class VariableViewBase;
  friend class NumArrayViewBase;

 private:

  AutoRef2<impl::RunQueueImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une commande associée à la file \a run_queue.
 */
inline RunCommand
makeCommand(const RunQueue& run_queue)
{
  run_queue._checkNotNull();
  return RunCommand(run_queue);
}

/*!
 * \brief Créé une commande associée à la file \a run_queue.
 */
inline RunCommand
makeCommand(const RunQueue* run_queue)
{
  ARCCORE_CHECK_POINTER(run_queue);
  run_queue->_checkNotNull();
  return RunCommand(*run_queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
