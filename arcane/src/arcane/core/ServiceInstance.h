// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceInstance.h                                           (C) 2000-2019 */
/*                                                                           */
/* Instance de service.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SERVICEINSTANCE_H
#define ARCANE_SERVICEINSTANCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"
#include "arcane/core/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Référence sur une instance de service.
 *
 * Cette classe est gérée via un compteur de référence à la manière
 * de la classe std::shared_ptr.
 */
class ARCANE_CORE_EXPORT ServiceInstanceRef
{
  typedef Ref<IServiceInstance> RefType;
 private:
  ServiceInstanceRef(const RefType& r) : m_instance(r){}
 public:
  ServiceInstanceRef() = default;
 public:
  static ServiceInstanceRef createRef(IServiceInstance* p)
  {
    return ServiceInstanceRef(RefType::create(p));
  }
  static ServiceInstanceRef createRefNoDestroy(IServiceInstance* p)
  {
    return ServiceInstanceRef(RefType::_createNoDestroy(p));
  }
  static ServiceInstanceRef createWithHandle(IServiceInstance* p,Internal::ExternalRef handle)
  {
    return ServiceInstanceRef(RefType::createWithHandle(p,handle));
  }
 public:
  IServiceInstance* get() const { return m_instance.get(); }
  void reset() { m_instance.reset(); }
 private:
  RefType m_instance;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool operator==(const ServiceInstanceRef& a,const ServiceInstanceRef& b)
{
  return a.get()==b.get();
}

inline bool operator!=(const ServiceInstanceRef& a,const ServiceInstanceRef& b)
{
  return a.get()!=b.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
