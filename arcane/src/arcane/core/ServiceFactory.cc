// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceFactory.cc                                           (C) 2000-2019 */
/*                                                                           */
/* Fabrique des services/modules.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"

#include "arcane/IServiceInfo.h"
#include "arcane/ServiceFactory.h"
#include "arcane/ServiceInstance.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ServiceFactory.h
 *
 * \brief Ce fichier contient les différentes fabriques de services
 * et macro pour enregistrer les services.
 *
 * La plupart des types de ce fichier sont internes à Arcane. Le seul élément
 * utile pour un utilisateur est la macro ARCANE_REGISTER_SERVICE() qui
 * permet d'enregistrer un service.
 */

/*!
 * \file ServiceProperty.h
 *
 * \brief Ce fichier contient les différentes types et classes
 * pour spécifier les propriétés d'un service.
 */


/*!
 * \defgroup Service Service
 *
 * \brief Ensemble des types utilisés dans la gestion des services.
 *
 * La plupart des services utilisateurs sont des services de
 * sous-domaine et dérivent indirectement de la classe
 * BasicService. En règle générale, un service est défini dans un
 * fichier AXL et l'outil \a axl2cc permet de générer la classe
 * de base d'un service à partir de ce fichier AXL. Pour plus
 * d'informations se reporter à la rubrique \ref arcanedoc_core_types_service.
 *
 * Il est néanmoins possible d'avoir des services sans fichier
 * AXL. Dans ce cas, l'enregistrement d'un service pour qu'il soit
 * reconnu par Arcane se fait via la macro ARCANE_REGISTER_SERVICE().
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Instances des services singletons.
 *
 * Les services singletons peuvent implémenter plusieurs interfaces.
 * Il y a donc une instance IServiceInstance par interface plus une instance
 * pour le service lui-même. Comme toutes ces instances référencent le
 * même service, il faut faire attention de ne détruire le service qu'une
 * seule fois.
 */
class SingletonServiceFactoryBase::ServiceInstance
: public ISingletonServiceInstance
, public IServiceInstanceAdder
{
 public:
  ServiceInstance(IServiceInfo* si)
  : m_service_info(si){}
  ~ServiceInstance()
  {
    destroyInstance();
  }
 public:
  void addReference() override { ++m_nb_ref; }
  void removeReference() override
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
    if (v==1)
      delete this;
  }
 public:
  ServiceInstanceCollection interfaceInstances() override { return m_instances; }
  void destroyInstance()
  {
    m_true_instance.reset();
    m_instances.clear();
  }
  IServiceInfo* serviceInfo() const override { return m_service_info; }
  void setTrueInstance(ServiceInstanceRef si) { m_true_instance = si; }
 public:
  void addInstance(ServiceInstanceRef instance) override
  {
    m_instances.add(instance);
  }
 private:
  IServiceInfo* m_service_info;
  List<ServiceInstanceRef> m_instances;
  ServiceInstanceRef m_true_instance;
  std::atomic<Int32> m_nb_ref = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé un service singleton
Ref<ISingletonServiceInstance> SingletonServiceFactoryBase::
createSingletonServiceInstance(const ServiceBuildInfoBase& sbib)
{
  auto x = new ServiceInstance(m_service_info);
  IServiceInstanceAdder* sia = x;
  ServiceInstanceRef si = _createInstance(sbib,sia);
  x->setTrueInstance(si);
  return makeRef<ISingletonServiceInstance>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractServiceFactory::
addReference()
{
  ++m_nb_ref;
}

void AbstractServiceFactory::
removeReference()
{
  // Décrémente et retourne la valeur d'avant.
  // Si elle vaut 1, cela signifie qu'on n'a plus de références
  // sur l'objet et qu'il faut le détruire.
  Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
  if (v==1)
    delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

