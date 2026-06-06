// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDirectSubDomainExecuteFunctor.h                            (C) 2000-2025 */
/*                                                                           */
/* Interface of a direct execution functor with subdomain.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDIRECTSUBDOMAINEXECUTEFUNCTOR_H
#define ARCANE_CORE_IDIRECTSUBDOMAINEXECUTEFUNCTOR_H
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
 * \internal
 * \brief Interface of a functor to execute code directly after
 * the creation of a subdomain without going through the time loop.
 */
class ARCANE_CORE_EXPORT IDirectSubDomainExecuteFunctor
{
 public:

  virtual ~IDirectSubDomainExecuteFunctor() = default;

 public:

  //! Executes the functor's operation
  virtual int execute() = 0;

  /*!
   * \brief Positions the associated subdomain.
   * This method must be called before execute()
   */
  virtual void setSubDomain(ISubDomain* sd) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
