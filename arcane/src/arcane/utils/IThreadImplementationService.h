// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IThreadImplementationService.h                              (C) 2000-2024 */
/*                                                                           */
/* Interface d'un service de gestion des threads.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ITHREADIMPLEMENTATIONSERVICE_H
#define ARCANE_UTILS_ITHREADIMPLEMENTATIONSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/IThreadImplementation.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service de gestion des threads.
 */
class ARCANE_UTILS_EXPORT IThreadImplementationService
{
 public:

  virtual ~IThreadImplementationService() = default;

 public:

  virtual Ref<IThreadImplementation> createImplementation() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
