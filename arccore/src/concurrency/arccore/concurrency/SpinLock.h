// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpinLock.h                                                  (C) 2000-2025 */
/*                                                                           */
/* SpinLock pour le multi-threading.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_SPINLOCK_H
#define ARCCORE_CONCURRENCY_SPINLOCK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/IThreadImplementation.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Ajouter énumération pour la gestion du spin_lock (None, FullSpin, Spin+Wait)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief SpinLock.
 */
class ARCCORE_CONCURRENCY_EXPORT SpinLock
{
 public:

  class ScopedLock
  {
   public:

    ScopedLock(SpinLock& sl)
    : m_spin_lock_ref(sl)
    {
      m_spin_lock_ref._doLock();
    }
    ~ScopedLock()
    {
      m_spin_lock_ref._doUnlock();
    }

   private:

    SpinLock& m_spin_lock_ref;
  };

  class ManualLock
  {
   public:

    void lock(SpinLock& sl)
    {
      sl._doLock();
    }
    void unlock(SpinLock& sl)
    {
      sl._doUnlock();
    }
  };

  friend class ScopedLock;
  friend class ManualLock;

 public:

  //! Mode du spinlock. Le défaut est 'Auto'
  enum class eMode : uint8_t
  {
    // Pas de synchronisation
    None,
    /*!
     * \brief Choix automatique.
     *
     * Si `Concurrency::getThreadImplementation()->isMultiThread()` est vrai,
     * alors le mode est SpinAndMutex. Sinon, le mode est None.
     */
    Auto,
    /*!
     * \brief Utilise toujours un spinlock.
     *
     * Ce type est plus rapide s'il y a très peu de contention mais les performances
     * sont très mauvaises dans le cas contraire.
     */
    FullSpin,
    /*!
     * \brief SpinLock puis mutex.
     *
     * Effectue un spinlock puis rend la main (std::this_thread::yield())
     * si cela dure trop longtemps. Ce mode n'est disponible que si on utilise le C++20.
     * Sinon, il est équivalent à FullSpin.
     */
    SpinAndMutex
  };

 public:

  //! SpinLock par défaut
  SpinLock();
  //! SpinLock avec le mode \a mode
  SpinLock(eMode mode);
  ~SpinLock();

 private:

  std::atomic_flag m_spin_lock = ATOMIC_FLAG_INIT;
  eMode m_mode = eMode::SpinAndMutex;

 private:

  void _doLock()
  {
    if (m_mode != eMode::None)
      _doLockReal();
  }
  void _doUnlock()
  {
    if (m_mode != eMode::None)
      _doUnlockReal();
  }

 private:

  // TODO: rendre ces fonctions inline lorsque le C++20 sera disponible
  // partout. Pour l'instant on ne peut pas le faire à cause de l'ODR.
  void _doLockReal();
  void _doUnlockReal();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
