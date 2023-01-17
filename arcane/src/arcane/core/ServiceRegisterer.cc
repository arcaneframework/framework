// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceRegisterer.cc                                        (C) 2000-2018 */
/*                                                                           */
/* Registre contenant la liste des manufactures de services.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"

#include "arcane/ServiceRegisterer.h"

#include <stdlib.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static ServiceRegisterer* global_arcane_first_service = nullptr;
static Integer global_arcane_nb_service = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceRegisterer::
ServiceRegisterer(ServiceInfoWithPropertyCreateFunc func,
                  const ServiceProperty& properties) ARCANE_NOEXCEPT
: m_module_factory_with_property_functor(nullptr)
, m_info_function_with_property(func)
, m_name(properties.name())
, m_service_property(properties)
, m_module_property(properties.name())
, m_previous(nullptr)
, m_next(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceRegisterer::
ServiceRegisterer(ModuleFactoryWithPropertyFunc func,
                  const ModuleProperty& properties) ARCANE_NOEXCEPT
: m_module_factory_with_property_functor(func)
, m_info_function_with_property(nullptr)
, m_name(properties.name())
, m_service_property(ServiceProperty(properties.name(),0))
, m_module_property(properties)
, m_previous(nullptr)
, m_next(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceRegisterer::
_init()
{
  if (global_arcane_first_service==nullptr){
    global_arcane_first_service = this;
    setPreviousService(nullptr);
    setNextService(nullptr);
  }
  else{
    ServiceRegisterer* next = global_arcane_first_service->nextService();
    setNextService(global_arcane_first_service); 
    global_arcane_first_service = this;
    if (next)
      next->setPreviousService(this);
  }
  ++global_arcane_nb_service;

  { // Check integrity
    ServiceRegisterer * p = global_arcane_first_service;
    Integer count = global_arcane_nb_service;
    while (p && count > 0) {
      p = p->nextService();
      --count;
    }
    if (p) {
      cout << "Arcane Fatal Error: Service '" << m_name << "' conflict in service registration" << std::endl;
      exit(1);
    } else if (count > 0) {
      cout << "Arcane Fatal Error: Service '" << m_name << "' breaks service registration (inconsistent shortcut)" << std::endl;
      exit(1);      
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceRegisterer* ServiceRegisterer::
firstService()
{
  return global_arcane_first_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ServiceRegisterer::
nbService()
{
  return global_arcane_nb_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

