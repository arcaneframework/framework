// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyBase.h                                           (C) 2000-2025 */
/*                                                                           */
/* Classes de base pour la gestion du multi-threading.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_CONCURRENCYBASE_H
#define ARCCORE_BASE_CONCURRENCYBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ParallelLoopOptions.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de base pour la gestion du multi-threading.
 */
class ARCCORE_BASE_EXPORT ConcurrencyBase
{
  // Pour appeler _setMaxAllowedThread.
  friend class TBBTaskImplementation;

 public:

  /*!
   * \brief Nombre maximum de threads autorisés pour le multi-threading.
   *
   * Cette valeur n'est significative qu'une fois que le service de gestion
   * du multi-threading a éte créé.
   */
  static Int32 maxAllowedThread() { return m_max_allowed_thread; }

 public:

  //! Positionne les valeurs par défaut d'exécution d'une boucle parallèle
  static void setDefaultParallelLoopOptions(const ParallelLoopOptions& v)
  {
    m_default_loop_options = v;
  }

  //! Valeurs par défaut d'exécution d'une boucle parallèle
  static const ParallelLoopOptions& defaultParallelLoopOptions()
  {
    return m_default_loop_options;
  }

 private:

  static Int32 m_max_allowed_thread;
  static ParallelLoopOptions m_default_loop_options;

 private:

  static void _setMaxAllowedThread(Int32 v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
