// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IObserver.h                                                 (C) 2000-2021 */
/*                                                                           */
/* Interface d'un observateur.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IOBSERVER_H
#define ARCANE_UTILS_IOBSERVER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IObservable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un observateur.
 *
 * Cette interface représente le concept d'observateur tel qu'il est
 * défini dans le Design Pattern.
 * Un observateur est attaché à un observable (IObservable) par la
 * méthode IObservable::attachObserver() et détaché par
 * IObservable::detachObserver(). L'observable le notifie d'un changement
 * en appelant la méthode observerUpdate().
 *
 * Un observateur ne peut-être attaché qu'à un seul observable à la fois
 *
 * Les méthodes de cette classe ne doivent être appelées que
 * par IObservable et jamais directement par l'utilisateur.
 *
 * \since 0.4.36
 * \author Gilles Grospellier
 * \date 25/06/2001
 *
 * \sa IObservable
 */
class ARCANE_UTILS_EXPORT IObserver
{
 protected:

  IObserver() {}

 public:

  virtual ~IObserver() {} //!< Libère les ressources

 public:

  //! \brief Notification venant de l'observable \a oba.
  virtual void observerUpdate(IObservable*) =0;

 public:

  //! S'attache à l'observable \a obs
  virtual void attachToObservable(IObservable* obs) =0;

  //! Se détache de l'observable
  virtual void detach() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

