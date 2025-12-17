// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimerMng.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Interface d'un gestionnaire de timer.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMERMNG_H
#define ARCANE_CORE_ITIMERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un gestionnaire de timer.
 *
 * Ce gestionnaire est utilisé exclusivement par les timers de
 * l'architecture (Timer) et ne doit pas être utilisé directement.
 *
 * Un timer utilise la méthode beginTimer() pour indiquer 
 * à ce gestionnaire qu'il souhaite débuter une mesure temporelle et
 * la méthode endTimer() pour indiquer que la mesure est terminée et
 * obtenir le temps écoulé depuis l'appel à beginTimer(). Il est aussi
 * possible d'obtenir le temps écoulé sans arrêter le timer par appel
 * à la fonction getTime().
 *
 * Les timers de même type s'imbriquent les uns dans les autres et doivent respecter
 * le principe des piles pour les appels à beginTimer() et endTimer(): le
 * timer qui appelle endTimer() doit être le dernier à avoir appelé beginTimer().
 *
 * Le type de temps utilisé est déterminé par le Timer::type(). Il s'agit soit du
 * temps CPU, soit du temps réel.
 *
 */
class ITimerMng
{
 public:

  /*!
   * \brief Libère les ressources.
   * \pre !hasTimer()
   */
  virtual ~ITimerMng() = default;

 public:

  /*!
   * \brief Attache le timer \a timer à ce gestionnaire.
   *
   * \pre !\a timer
   * \pre !hasTimer(\a timer)
   * \post hasTimer(\a timer)
   */
  virtual void beginTimer(Timer* timer) = 0;

  /*!
   * \brief Relâche le timer \a timer.
   *
   * \return le temps écoulé depuis l'appel à beginTimer().
   *
   * \pre !\a timer
   * \pre hasTimer(\a timer)
   * \post !hasTimer(\a timer)
   */
  virtual Real endTimer(Timer* timer) = 0;

  /*!
   * \brief Temps écoulé depuis le dernier appel à beginTimer().
   *
   * \pre !\a timer
   * \pre hasTimer(\a timer)
   */
  virtual Real getTime(Timer* timer) = 0;

  /*!
   * \brief Indique si le timer \a timer est enregistré.
   *
   * \pre !\a timer
   * \deprecated Cette fonction sera supprimé à terme. Ne plus utiliser.
   */
  ARCCORE_DEPRECATED_2019("Do not use this method")
  virtual bool hasTimer(Timer* timer) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
