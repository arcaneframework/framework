// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TaskFactoryInternal.h                                       (C) 2000-2025 */
/*                                                                           */
/* API interne à Arcane de 'TaskFactory'.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_TASKFACTORYINTERNAL_H
#define ARCANE_UTILS_INTERNAL_TASKFACTORYINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API interne à Arcane de 'TaskFactory'.
 */
class ARCANE_UTILS_EXPORT TaskFactoryInternal
{
 public:

  //! Ajoute un observateur pour la création de thread.
  static void addThreadCreateObserver(IObserver* o);

  //! Supprime un observateur pour la création de thread.
  static void removeThreadCreateObserver(IObserver* o);

  //! Notifie tous les observateurs de création de thread
  static void notifyThreadCreated();

 public:

  static void setImplementation(ITaskImplementation* task_impl);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
