// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdThreadImplementation.cc                                  (C) 2000-2024 */
/*                                                                           */
/* Implémentation des threads utilisant la bibliothèque standard C++.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/IThreadImplementationService.h"
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
