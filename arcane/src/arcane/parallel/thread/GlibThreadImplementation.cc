// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GlibThreadImplementation.cc                                 (C) 2000-2018 */
/*                                                                           */
/* Implémentation des threads utilisant la glib.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/GlibThreadImplementation.h"

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/IThreadImplementationService.h"
#include "arcane/FactoryService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation des threads utilisant Glib.
 */
class ArcaneGlibThreadImplementation
: public Arccore::GlibThreadImplementation
{
 public:
  ArcaneGlibThreadImplementation()
  {
  }

 public:

  void build(){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GlibThreadImplementationService
: public IThreadImplementationService
{
 public:
  GlibThreadImplementationService(const ServiceBuildInfo&){}
 public:
  void build() {}
 public:
  IThreadImplementation* createImplementation() override
  {
    return new Arccore::GlibThreadImplementation();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(GlibThreadImplementationService,
                                    IThreadImplementationService,
                                    GlibThreadImplementationService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
