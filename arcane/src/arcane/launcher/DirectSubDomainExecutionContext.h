// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DirectSubDomainExecutionContext.h                           (C) 2000-2021 */
/*                                                                           */
/* Direct execution context with subdomain creation.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_DIRECTSUBDOMAINEXECUTIONCONTEXT_H
#define ARCANE_LAUNCHER_DIRECTSUBDOMAINEXECUTIONCONTEXT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISubDomain;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Direct execution context with subdomain creation.
 */
class ARCANE_LAUNCHER_EXPORT DirectSubDomainExecutionContext
{
  class Impl;
  friend class ArcaneLauncherDirectExecuteFunctor;

 protected:

  // Protected method so that an instance can only be created via Arcane
  DirectSubDomainExecutionContext(ISubDomain* sd);

 public:

  ~DirectSubDomainExecutionContext();
  DirectSubDomainExecutionContext(const DirectSubDomainExecutionContext&) = delete;
  DirectSubDomainExecutionContext& operator=(const DirectSubDomainExecutionContext&) = delete;

 public:

  //! Sub domain
  ISubDomain* subDomain() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
