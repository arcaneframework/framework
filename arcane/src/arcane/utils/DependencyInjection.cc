// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DependencyInjection.cc                                      (C) 2000-2021 */
/*                                                                           */
/* Types et fonctions pour gérer le pattern 'DependencyInjection'.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/DependencyInjection.h"

#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::DependencyInjection
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Injector::Impl
{
 public:
  ~Impl()
  {
    for( Integer i=0, n= m_instance_list.size(); i<n; ++i )
      delete m_instance_list[i];
    m_instance_list.clear();
  }
 public:
  UniqueArray<IInjectedInstance*> m_instance_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Injector::
Injector()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Injector::
_add(IInjectedInstance* instance)
{
  m_p->m_instance_list.add(instance);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Injector::
_nbValue() const
{
  return m_p->m_instance_list.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IInjectedInstance* Injector::
_value(Integer i) const
{
  return m_p->m_instance_list[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GlobalRegisterer::
GlobalRegisterer(FactoryCreateFunc func,const ProviderProperty& property)  ARCANE_NOEXCEPT
: m_factory_create_func(func)
, m_factory_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
GlobalRegisterer* global_arcane_first_service = nullptr;
Integer global_arcane_nb_service = 0;
}

void GlobalRegisterer::
_init()
{
  // ATTENTION: Cette méthode est appelée depuis un constructeur global
  // (donc avant le main()) et il ne faut pas faire d'exception dans ce code.
  if (!global_arcane_first_service){
    global_arcane_first_service = this;
    _setPreviousService(nullptr);
    _setNextService(nullptr);
  }
  else{
    GlobalRegisterer* next = global_arcane_first_service->nextService();
    _setNextService(global_arcane_first_service);
    global_arcane_first_service = this;
    if (next)
      next->_setPreviousService(this);
  }
  ++global_arcane_nb_service;

  {
    // Check integrity
    GlobalRegisterer * p = global_arcane_first_service;
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

GlobalRegisterer* GlobalRegisterer::
firstService()
{
  return global_arcane_first_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer GlobalRegisterer::
nbService()
{
  return global_arcane_nb_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::DependencyInjection

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
