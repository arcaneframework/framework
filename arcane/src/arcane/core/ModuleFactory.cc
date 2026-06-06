// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleFactory.cc                                            (C) 2000-2019 */
/*                                                                           */
/* Module manufacturing.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/core/ModuleFactory.h"
#include "arcane/core/IModuleMng.h"
#include "arcane/core/IModule.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IServiceInfo.h"
#include "arcane/core/IMesh.h"

#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleFactory::
ModuleFactory(Ref<IModuleFactory2> factory, bool is_autoload)
: m_factory(factory)
, m_is_autoload(is_autoload)
, m_name(factory->moduleName())
, m_nb_ref(0)
{
  //cerr << "** ADD MODULE FACTORY this=" << this
  //     << " service_info_name=" << m_service_info->localName()
  //     << " autoload=" << is_autoload << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleFactory::
~ModuleFactory()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IModule> ModuleFactory::
createModule(ISubDomain* parent, const MeshHandle& mesh_handle)
{
  if (!m_factory)
    ARCANE_FATAL("Null factory for module named '{0}'", moduleName());

  Ref<IModule> module = m_factory->createModuleInstance(parent, mesh_handle);

  if (!module)
    ARCANE_FATAL("Can not create module named '{0}'", moduleName());

  parent->checkId("ModuleFactory::createModule", module->name());
  parent->moduleMng()->addModule(module);

  return module;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleFactory::
initializeModuleFactory(ISubDomain* sub_domain)
{
  m_factory->initializeModuleFactory(sub_domain);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const IServiceInfo* ModuleFactory::
serviceInfo() const
{
  return m_factory->serviceInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleFactory::
addReference()
{
  ++m_nb_ref;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleFactory::
removeReference()
{
  // Decrements and returns the previous value.
  // If it equals 1, it means there are no more references
  // to the object and it must be destroyed.
  Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
  if (v == 1)
    delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleFactoryReference::
ModuleFactoryReference(Ref<IModuleFactory2> factory, bool is_autoload)
: Base(new ModuleFactory(factory, is_autoload))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleFactory2::
~ModuleFactory2()
{
  if (m_service_info) {
    delete m_service_info->factoryInfo();
    delete m_service_info;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
