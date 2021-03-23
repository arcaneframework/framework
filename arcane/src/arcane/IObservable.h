// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IObservable.h                                               (C) 2000-2016 */
/*                                                                           */
/* Interface d'un observable.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IOBSERVABLE_H
#define ARCANE_IOBSERVABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IObserver;

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
class ARCANE_CORE_EXPORT IObservable
{
 public:

  virtual ~IObservable() {} //!< Libère les ressources

 public:

  static IObservable* createDefault();

 public:

  /*!
   * \brief Détruit l'observable.
   * Cet appel détruit l'instance (via delete this). Elle
   * ne doit donc plus être utilisé par la suite.
   *
   * \deprecated Utiliser detachAllObservers() puis l'opérateur delete ensuite.
   */
  virtual ARCANE_DEPRECATED_220 void destroy() =0;

  /*!
   * \brief Attache l'observateur \a obs à cette observable.
   *
   * Il est possible d'attacher un observateur plus d'une fois.
   */
  virtual void attachObserver(IObserver* obs) =0;

  /*!
   * \brief Détache l'observateur \a obs de cette observable.
   *
   * Si l'obervateur \a obs n'est pas présent, rien n'est fait. S'il est
   * présent plusieurs fois, c'est la dernière occurence qui est effacé.
   */
  virtual void detachObserver(IObserver* obs) =0;

  /*!
   * \brief Notifie tous les observateurs.
   *
   * Pour chaque observateur attaché, appelle IObserver::observerUpdate().
   */
  virtual void notifyAllObservers() =0;

  //! Vrai si des observers sont attachées à cette observable.
  virtual bool hasObservers() const =0;

  /*!
   * \brief Vrai si l'observable est détruit et ne doit plus être utilisé.
   *
   */
  virtual ARCANE_DEPRECATED_220 bool isDestroyed() const =0;

  /*!
   * \brief Détache tous les observeurs associés à cette instance.
   */
  virtual void detachAllObservers() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

