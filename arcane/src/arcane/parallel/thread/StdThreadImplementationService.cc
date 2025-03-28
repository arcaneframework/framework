// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdThreadImplementationService.cc                           (C) 2000-2025 */
/*                                                                           */
/* Implémentation des threads utilisant la bibliothèque standard C++.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/IThreadImplementationService.h"
#include "arcane/utils/internal/DependencyInjection.h"

#include "arcane/core/FactoryService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StdThreadImplementationService
: public IThreadImplementationService
{
 public:

  explicit StdThreadImplementationService(const ServiceBuildInfo&) {}
  StdThreadImplementationService() = default;

 public:

  void build() {}

 public:

  Ref<IThreadImplementation> createImplementation() override
  {
    return Arccore::Concurrency::createStdThreadImplementation();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(StdThreadImplementationService,
                                    IThreadImplementationService,
                                    StdThreadImplementationService);

ARCANE_DI_REGISTER_PROVIDER(StdThreadImplementationService,
                            DependencyInjection::ProviderProperty("StdThreadImplementationService"),
                            ARCANE_DI_INTERFACES(IThreadImplementationService),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
