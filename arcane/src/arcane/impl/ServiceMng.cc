// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceMng.cc                                               (C) 2000-2013 */
/*                                                                           */
/* Classe gérant l'ensemble des services.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/List.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Ref.h"
#include "arccore/base/ReferenceCounter.h"

#include "arcane/IBase.h"
#include "arcane/IServiceMng.h"
#include "arcane/IService.h"
#include "arcane/IServiceInfo.h"
#include "arcane/ServiceInstance.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des services.
 */
class ServiceMng
: public IServiceMng
{
 public:

  ServiceMng(IBase*);
  ~ServiceMng();

 public:
  
  ITraceMng* traceMng() const override { return m_base->traceMng(); }

  void addSingletonInstance(SingletonServiceInstanceRef sv) override
  {
    m_singleton_instances.add(sv);
  }

  SingletonServiceInstanceCollection singletonServices() const override
  {
    return m_singleton_instances;
  }

  SingletonServiceInstanceRef singletonServiceReference(const String& name) const override;

 private:

  IBase* m_base; //!< Gestionnaire principal
  List<SingletonServiceInstanceRef> m_singleton_instances; //!< Liste des instances singletons

 private:

  void onServicesChanged(const CollectionEventArgs& args);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IServiceMng*
arcaneCreateServiceMng(IBase* b)
{
  return new ServiceMng(b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceMng::
ServiceMng(IBase* b)
: m_base(b)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceMng::
~ServiceMng()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SingletonServiceInstanceRef ServiceMng::
singletonServiceReference(const String& name) const
{
  for( const SingletonServiceInstanceRef& sr : m_singleton_instances ){
    IServiceInstance* si = sr.get();
    if (si){
      IServiceInfo* sii = si->serviceInfo();
      if (sii && sii->localName()==name)
        return sr;
    }
  }
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceMng::
onServicesChanged(const CollectionEventArgs& args)
{
  ARCANE_UNUSED(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

