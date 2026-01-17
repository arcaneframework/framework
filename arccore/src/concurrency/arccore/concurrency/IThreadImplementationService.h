// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IThreadImplementationService.h                              (C) 2000-2026 */
/*                                                                           */
/* Interface d'un service de gestion des threads.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_ITHREADIMPLEMENTATIONSERVICE_H
#define ARCCORE_CONCURRENCY_ITHREADIMPLEMENTATIONSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service de gestion des threads.
 */
class ARCCORE_CONCURRENCY_EXPORT IThreadImplementationService
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
