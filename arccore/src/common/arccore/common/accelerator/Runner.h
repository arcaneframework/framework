// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Gestion de l'exécution sur accélérateur.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNNER_H
#define ARCCORE_COMMON_ACCELERATOR_RUNNER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Ref.h"

#include "arccore/common/accelerator/RunQueue.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire d'exécution pour accélérateur.
 *
 * Cette classe utilise une sémantique par référence
 *
 * Une instance de cette classe représente un back-end d'exécution. Il faut
 * d'abord appelé initialize() avant de pouvoir utiliser les méthodes de
 * l'instance ou alors il faut appeler l'un des constructeurs autre que le
 * constructeur par défaut. Le back-end utilisé est choisi via l'énumération
 * eExecutionPolicy. Les back-ends sont de deux types:
 * - les back-ends qui s'exécutent sur l'hôte: eExecutionPolicy::Sequential
 * et eExecutionPolicy::Thread,
 * - les back-ends qui s'exécutent sur accélérateurs : eExecutionPolicy::CUDA,
 * eExecutionPolicy::HIP et eExecutionPolicy::SYCL.
 *
 * La fonction \arcaneacc{isAcceleratorPolicy()} permet de savoir si une
 * eExecutionPolicy est associée à un accélérateur.
 *
 * Si une instance de cette classe est associée à un accélérateur, celui-ci
 * n'est pas forcément celui utilisé par défaut pour le thread courant.
 * Pour garantir que les kernels associés à ce runner seront bien exécutés
 * sur le bon device il est nécessaire d'appeler au moins une fois
 * la méthode setAsCurrentDevice() et de le refaire si une autre partie du code
 * ou si une bibliothèque externe change l'accélérateur par défaut.
 *
 * La classe Runner permet de créer des files d'exécutions (RunQueue)
 * via la fonction makeQueue(). Ces files peuvent ensuite être utilisées
 * pour lancer des commandes (RunCommand). La page \ref arcanedoc_acceleratorapi
 * décrit le fonctionnement de l'API accélérateur.
 */
class ARCCORE_COMMON_EXPORT Runner
{
  friend impl::RunQueueImpl;
  friend impl::RunCommandImpl;
  friend RunQueue;
  friend RunQueueEvent;
  friend impl::RunnerImpl;

  friend RunQueue makeQueue(const Runner& runner);
  friend RunQueue makeQueue(const Runner* runner);
  friend RunQueue makeQueue(const Runner& runner, const RunQueueBuildInfo& bi);
  friend RunQueue makeQueue(const Runner* runner, const RunQueueBuildInfo& bi);
  friend Ref<RunQueue> makeQueueRef(const Runner& runner);
  friend Ref<RunQueue> makeQueueRef(Runner& runner, const RunQueueBuildInfo& bi);
  friend Ref<RunQueue> makeQueueRef(Runner* runner);

 public:

  /*!
   * \brief Créé un gestionnaire d'exécution non initialisé.
   *
   * Il faudra appeler initialize() avant de pouvoir utiliser l'instance
   */
  Runner();
  //! Créé et initialise un gestionnaire pour l'accélérateur \a p
  explicit Runner(eExecutionPolicy p);
  //! Créé et initialise un gestionnaire pour l'accélérateur \a p et l'accélérateur \a device
  Runner(eExecutionPolicy p, DeviceId device);

 public:

  //! Politique d'exécution associée
  eExecutionPolicy executionPolicy() const;

  //! Initialise l'instance. Cette méthode ne doit être appelée qu'une seule fois.
  void initialize(eExecutionPolicy v);

  //! Initialise l'instance. Cette méthode ne doit être appelée qu'une seule fois.
  void initialize(eExecutionPolicy v, DeviceId device);

  //! Indique si l'instance a été initialisée
  bool isInitialized() const;

  /*!
   * \brief Indique si on autorise la création de RunQueue depuis plusieurs threads.
   *
   * \deprecated La création de file est toujours thread-safe depuis la version
   * 3.15 de Arcane.
   */
  ARCCORE_DEPRECATED_REASON("Y2025: this method is a no op. Concurrent queue creation is always thread-safe")
  void setConcurrentQueueCreation(bool v);

  //! Indique si la création concurrent de plusieurs RunQueue est autorisé
  bool isConcurrentQueueCreation() const;

  /*!
   * \brief Temps total passé dans les commandes associées à cette instance.
   *
   * Ce temps n'est significatif que si les RunQueue sont synchrones.
   */
  double cumulativeCommandTime() const;

  //! Positionne la politique d'exécution des réductions
  ARCCORE_DEPRECATED_REASON("Y2025: this method is a no op. reduce policy is always eDeviceReducePolicy::Grid")
  void setDeviceReducePolicy(eDeviceReducePolicy v);

  //! politique d'exécution des réductions
  eDeviceReducePolicy deviceReducePolicy() const;

  //! Positionne un conseil sur la gestion d'une zone mémoire
  void setMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice);

  //! Supprime un conseil sur la gestion d'une zone mémoire
  void unsetMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice);

  //! Device associé à cette instance.
  DeviceId deviceId() const;

  /*!
   * \brief Positionne le device associé à cette instance comme le device par défaut du contexte.
   *
   * Cet appel est équivalent à cudaSetDevice() ou hipSetDevice();
   */
  void setAsCurrentDevice();

  //! Information sur le device associé à cette instance.
  const DeviceInfo& deviceInfo() const;

  //! Information sur le device associé à cette instance.
  DeviceMemoryInfo deviceMemoryInfo() const;

  //! Remplit \a attr avec les informations concernant la zone mémoire pointée par \a ptr
  void fillPointerAttribute(PointerAttribute& attr, const void* ptr);

 public:

  /*!
   * \brief Liste des devices pour la politique d'exécution \a policy.
   *
   * Si le runtime associé n'a pas encore été initialisé, cette méthode retourne \a nullptr.
   */
  static const IDeviceInfoList* deviceInfoList(eExecutionPolicy policy);

 private:

  // La création est réservée aux méthodes globales makeQueue()
  static RunQueue _makeQueue(const Runner& runner)
  {
    return RunQueue(runner, true);
  }
  static RunQueue _makeQueue(const Runner& runner, const RunQueueBuildInfo& bi)
  {
    return RunQueue(runner, bi, true);
  }
  static Ref<RunQueue> _makeQueueRef(const Runner& runner)
  {
    return makeRef(new RunQueue(runner, true));
  }
  static Ref<RunQueue> _makeQueueRef(Runner& runner, const RunQueueBuildInfo& bi)
  {
    return makeRef(new RunQueue(runner, bi, true));
  }

 public:

  //! API interne à %Arcane
  RunnerInternal* _internalApi();

 private:

  impl::IRunnerRuntime* _internalRuntime() const;
  impl::RunnerImpl* _impl() const { return m_p.get(); }

 private:

  std::shared_ptr<impl::RunnerImpl> m_p;

 private:

  void _checkIsInit() const;
  bool _isAutoPrefetchCommand() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Créé une file associée à \a runner.
 *
 * Cet appel est thread-safe.
 */
inline RunQueue
makeQueue(const Runner& runner)
{
  return Runner::_makeQueue(runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une file associée à \a runner.
 *
 * Cet appel est thread-safe.
 */
inline RunQueue
makeQueue(const Runner* runner)
{
  ARCCORE_CHECK_POINTER(runner);
  return Runner::_makeQueue(*runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une file associée à \a runner avec les propriétés \a bi.
 *
 * Cet appel est thread-safe.
 */
inline RunQueue
makeQueue(const Runner& runner, const RunQueueBuildInfo& bi)
{
  return Runner::_makeQueue(runner, bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une file associée à \a runner avec les propriétés \a bi.
 *
 * Cet appel est thread-safe.
 */
inline RunQueue
makeQueue(const Runner* runner, const RunQueueBuildInfo& bi)
{
  ARCCORE_CHECK_POINTER(runner);
  return Runner::_makeQueue(*runner, bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une référence sur file avec la politique d'exécution par défaut de \a runner.
 *
 * Si la file est temporaire, il est préférable d'utiliser makeQueue() à la place
 * pour éviter une allocation inutile.
 */
inline Ref<RunQueue>
makeQueueRef(const Runner& runner)
{
  return Runner::_makeQueueRef(runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une référence sur file avec la politique d'exécution par défaut de \a runner.
 *
 * Si la file est temporaire, il est préférable d'utiliser makeQueue() à la place
 * pour éviter une allocation inutile.
 */
inline Ref<RunQueue>
makeQueueRef(Runner& runner, const RunQueueBuildInfo& bi)
{
  return Runner::_makeQueueRef(runner, bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une référence sur file avec la politique d'exécution par défaut de \a runner.
 *
 * Si la file est temporaire, il est préférable d'utiliser makeQueue() à la place
 * pour éviter une allocation inutile.
 */
inline Ref<RunQueue>
makeQueueRef(Runner* runner)
{
  ARCCORE_CHECK_POINTER(runner);
  return Runner::_makeQueueRef(*runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
