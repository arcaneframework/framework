// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IObservable.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface d'un observable.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_IOBSERVABLE_H
#define ARCCORE_BASE_IOBSERVABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Interface d'un observable.
 *
 * Un observable est un objet qui maintient une liste d'observateurs
 * (IObserver) et qui permet de les notifier d'un évènement par l'intermédiaire
 * de la méthode notifyAllObserver().
 *
 * Un observateur est ajouté à la liste des observateurs par la méthode
 * attachObserver() et supprimé de cette même liste par detachObserver().
 *
 * La liste des observateurs attachés est ordonnée et les notifications se font
 * dans l'ordre des éléments de la liste. Si un même observateur est présent
 * plusieurs fois, il sera notifier autant de fois qu'il est présent.
 *
 * \warning Il est indispensable de supprimer, via l'appel à detachAllObservers()
 * les observeurs associés à un observable avant de le détruire.
 *
 * \sa IObserver
 */
class ARCCORE_BASE_EXPORT IObservable
{
 public:

  virtual ~IObservable() {} //!< Libère les ressources

 public:

  static IObservable* createDefault();

 public:

  /*!
   * \brief Attache l'observateur \a obs à cette observable.
   *
   * Il est possible d'attacher un observateur plus d'une fois.
   */
  virtual void attachObserver(IObserver* obs) = 0;

  /*!
   * \brief Détache l'observateur \a obs de cette observable.
   *
   * Si l'obervateur \a obs n'est pas présent, rien n'est fait. S'il est
   * présent plusieurs fois, c'est la dernière occurence qui est effacé.
   */
  virtual void detachObserver(IObserver* obs) = 0;

  /*!
   * \brief Notifie tous les observateurs.
   *
   * Pour chaque observateur attaché, appelle IObserver::observerUpdate().
   */
  virtual void notifyAllObservers() = 0;

  //! Vrai si des observers sont attachées à cette observable.
  virtual bool hasObservers() const = 0;

  /*!
   * \brief Détache tous les observeurs associés à cette instance.
   */
  virtual void detachAllObservers() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

