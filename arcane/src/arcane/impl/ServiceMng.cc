// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceMng.cc                                               (C) 2000-2013 */
/*                                                                           */
/* Class managing all services.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/List.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Ref.h"
#include "arccore/base/ReferenceCounter.h"

#include "arcane/core/IBase.h"
#include "arcane/core/IServiceMng.h"
#include "arcane/core/IService.h"
#include "arcane/core/IServiceInfo.h"
#include "arcane/core/ServiceInstance.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service manager.
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

  IBase* m_base; //!< Main manager
  List<SingletonServiceInstanceRef> m_singleton_instances; //!< List of singleton instances

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
  for (const SingletonServiceInstanceRef& sr : m_singleton_instances) {
    IServiceInstance* si = sr.get();
    if (si) {
      IServiceInfo* sii = si->serviceInfo();
      if (sii && sii->localName() == name)
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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
