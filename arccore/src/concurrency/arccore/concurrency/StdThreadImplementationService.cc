// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdThreadImplementationService.cc                           (C) 2000-2026 */
/*                                                                           */
/* Implementation of threads using the C++ standard library.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/internal/DependencyInjection.h"

#include "arccore/concurrency/internal/ConcurrencyGlobalInternal.h"
#include "arccore/concurrency/IThreadImplementation.h"
#include "arccore/concurrency/IThreadImplementationService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation of threads using C++20 STL classes.
 */
class StdThreadImplementationService
: public IThreadImplementationService
{
 public:

  void build() {}

 public:

  Ref<IThreadImplementation> createImplementation() override
  {
    return Concurrency::createStdThreadImplementation();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation of threads using LegacyStdBarrier for barriers.
 *
 * This implementation is only useful while the StdThreadImplementation
 * is being validated (March 2026). It may be removed later.
 */
class LegacyStdThreadImplementationService
: public IThreadImplementationService
{
 public:

  void build() {}

 public:

  Ref<IThreadImplementation> createImplementation() override
  {
    return Concurrency::createLegacyStdThreadImplementation();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DI_REGISTER_PROVIDER(StdThreadImplementationService,
                            DependencyInjection::ProviderProperty("StdThreadImplementationService"),
                            ARCANE_DI_INTERFACES(IThreadImplementationService),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());

ARCANE_DI_REGISTER_PROVIDER(LegacyStdThreadImplementationService,
                            DependencyInjection::ProviderProperty("LegacyStdThreadImplementationService"),
                            ARCANE_DI_INTERFACES(IThreadImplementationService),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
