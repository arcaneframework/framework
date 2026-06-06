// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionService.h                                         (C) 2000-2025 */
/*                                                                           */
/* Data set options using a service.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEOPTIONSERVICE_H
#define ARCANE_CORE_CASEOPTIONSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Functor.h"

#include "arcane/core/CaseOptions.h"
#include "arcane/core/IServiceMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ServiceUtils.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/IFactoryService.h"
#include "arcane/core/IServiceFactory.h"
#include "arcane/core/StringDictionary.h"
#include "arcane/core/CaseOptionServiceImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IService;
class CaseOptionBuildInfo;
template<typename T> class CaseOptionServiceT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation of the container for a service of type \a InterfaceType.
 */
template<typename InterfaceType>
class CaseOptionServiceContainer
: public ICaseOptionServiceContainer
{
 public:
  ~CaseOptionServiceContainer() override
  {
    removeInstances();
  }

  bool tryCreateService(Integer index,Internal::IServiceFactory2* factory,const ServiceBuildInfoBase& sbi) override
  {
    auto true_factory = dynamic_cast< Internal::IServiceFactory2T<InterfaceType>* >(factory);
    if (true_factory){
      Ref<InterfaceType> sr = true_factory->createServiceReference(sbi);
      InterfaceType* s = sr.get();
      m_services_reference[index] = sr;
      m_services[index] = s;
      return (s != nullptr);
    }
    return false;
  }

  bool hasInterfaceImplemented(Internal::IServiceFactory2* factory) const override
  {
    auto true_factory = dynamic_cast< Internal::IServiceFactory2T<InterfaceType>* >(factory);
    if (true_factory){
      return true;
    }
    return false;
  }

  //! Allocates an array for \a size elements
  void allocate(Integer asize) override
  {
    m_services.resize(asize,nullptr);
    m_services_reference.resize(asize);
  }

  //! Returns the number of elements in the array.
  Integer nbElem() const override
  {
    return m_services.size();
  }

  InterfaceType* child(Integer i) const
  {
    return m_services[i];
  }

  Ref<InterfaceType> childRef(Integer i) const
  {
    return m_services_reference[i];
  }

 public:
  //! Removes service instances
  void removeInstances()
  {
    m_services_reference.clear();
    m_services.clear();
  }
 public:
  ArrayView<InterfaceType*> view() { return m_services; }
 private:
  UniqueArray<InterfaceType*> m_services;
  UniqueArray<Ref<InterfaceType>> m_services_reference;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup CaseOption
 * \brief Base class for options using services.
 *
 * Instances of this class are not copyable.
 */
class ARCANE_CORE_EXPORT CaseOptionService
{
 public:

 CaseOptionService(const CaseOptionBuildInfo& cob,bool allow_null,bool is_optional)
  : m_impl(new CaseOptionServiceImpl(cob,allow_null,is_optional))
  {
  }

  virtual ~CaseOptionService() = default;

 public:

  CaseOptionService(const CaseOptionService&) = delete;
  const CaseOptionService& operator=(const CaseOptionService&) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use toICaseOptions() instead")
  operator CaseOptions& () { return *_impl(); }

  ARCANE_DEPRECATED_REASON("Y2022: Use toICaseOptions() instead")
  operator const CaseOptions& () const { return *_impl(); }

 public:

   const ICaseOptions* toICaseOptions() { return _impl(); }

 public:

  String rootTagName() const { return m_impl->rootTagName(); }
  String name() const { return m_impl->name(); }
  String serviceName() const { return m_impl->serviceName(); }
  bool isOptional() const { return m_impl->isOptional(); }
  bool isPresent() const { return m_impl->isPresent(); }
  void addAlternativeNodeName(const String& lang,const String& name)
  {
    m_impl->addAlternativeNodeName(lang,name);
  }
  void getAvailableNames(StringArray& names) const
  {
    m_impl->getAvailableNames(names);
  }

  /*!
   * \brief Sets the default value for the service name.
   *
   * If the option is not present in the data set, its value will be
   * that specified by the \a def_value argument; otherwise, calling this method
   * has no effect.
   *
   * This method can only be called during phase 1 of reading
   * the data set because the service is already instantiated afterwards.
   * A FatalErrorException exception is raised if this method is called and the
   * service is already instantiated.
   */
  void setDefaultValue(const String& def_value)
  {
    m_impl->setDefaultValue(def_value);
  }

  //! Adds the default value \a value to the category \a category
  void addDefaultValue(const String& category,const String& value)
  {
    m_impl->addDefaultValue(category,value);
  }

  /*!
   * \brief Sets the mesh name to which the service will be associated.
   *
   * If null, the service is associated with the default mesh of the subdomain
   * (ISubDomain::defaultMeshHandle()). The actual association happens when reading
   * the options. Calling this method after reading the options will have
   * no impact.
   */
  void setMeshName(const String& mesh_name);

  /*!
   * \brief Mesh name to which the service is associated.
   *
   * This is the name of the mesh as specified in the service descriptor
   * (the 'axl' file). To get the associated mesh after reading the options
   * you must use ICaseOptions::meshHandle().
   */
  String meshName() const;

 protected:

  CaseOptionServiceImpl* _impl() { return m_impl.get(); }
  const CaseOptionServiceImpl* _impl() const { return m_impl.get(); }

 private:

  ReferenceCounter<CaseOptionServiceImpl> m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class InterfaceType>
class CaseOptionServiceT
: public CaseOptionService
{
 public:
  CaseOptionServiceT(const CaseOptionBuildInfo& cob,bool allow_null,bool is_optional)
  : CaseOptionService(cob,allow_null,is_optional)
  {
    _impl()->setContainer(&m_container);
  }
  ~CaseOptionServiceT() = default;
 public:
  InterfaceType* operator()() const { return _instance(); }
  InterfaceType* instance() const { return _instance(); }
  Ref<InterfaceType> instanceRef() const { return _instanceRef(); }
 private:
  CaseOptionServiceContainer<InterfaceType> m_container;
 private:
  InterfaceType* _instance() const
  {
    if (m_container.nbElem()==1)
      return m_container.child(0);
    return nullptr;
  }
  Ref<InterfaceType> _instanceRef() const
  {
    if (m_container.nbElem()==1)
      return m_container.childRef(0);
    return {};
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup CaseOption
 * \brief Base class for a service option that can appear multiple times.
 */
class ARCANE_CORE_EXPORT CaseOptionMultiService
{
 public:
  CaseOptionMultiService(const CaseOptionBuildInfo& cob,bool allow_null)
  : m_impl(new CaseOptionMultiServiceImpl(cob,allow_null))
  {
  }
  virtual ~CaseOptionMultiService() = default;
  CaseOptionMultiService(const CaseOptionMultiService&) = delete;
  const CaseOptionMultiService& operator=(const CaseOptionMultiService&) = delete;
 public:
  XmlNode rootElement() const { return m_impl->toCaseOptions()->configList()->rootElement(); }
  String rootTagName() const { return m_impl->rootTagName(); }
  String name() const { return m_impl->name(); }
  //! Returns the valid implementation names for this service in \a names
  void getAvailableNames(StringArray& names) const
  {
    m_impl->getAvailableNames(names);
  }
  //! Name of the n-th service
  String serviceName(Integer index) const
  {
    return m_impl->serviceName(index);
  }
  void addAlternativeNodeName(const String& lang,const String& name)
  {
    m_impl->addAlternativeNodeName(lang,name);
  }
  /*!
   * \brief Sets the mesh name to which the service will be associated.
   *
   * \sa CaseOptionService::setMeshName()
   */
  void setMeshName(const String& mesh_name);

  /*!
   * \brief Mesh name to which the service is associated.
   *
   * \sa CaseOptionService::axlMeshName();
   */
  String meshName() const;

 protected:

  CaseOptionMultiServiceImpl* _impl() { return m_impl.get(); }
  const CaseOptionMultiServiceImpl* _impl() const { return m_impl.get(); }

 private:

  ReferenceCounter<CaseOptionMultiServiceImpl> m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup CaseOption
 * \brief Data set option of the service list type.
 */
template<typename InterfaceType>
class CaseOptionMultiServiceT
: public CaseOptionMultiService
, public ArrayView<InterfaceType*>
{
  typedef CaseOptionMultiServiceT<InterfaceType> ThatClass;
 public:
  CaseOptionMultiServiceT(const CaseOptionBuildInfo& cob,bool allow_null)
  : CaseOptionMultiService(cob,allow_null)
  , m_notify_functor(this,&ThatClass::_notify)
  {
    _impl()->setContainer(&m_container);
    _impl()->_setNotifyAllocateFunctor(&m_notify_functor);
  }
 public:
  CaseOptionMultiServiceT<InterfaceType>& operator()()
  {
    return *this;
  }
  const CaseOptionMultiServiceT<InterfaceType>& operator()() const
  {
    return *this;
  }
 protected:
  // Notification by the implementation
  void _notify()
  {
    this->setArray(m_container.view());
  }
 private:
  CaseOptionServiceContainer<InterfaceType> m_container;
  FunctorT<ThatClass> m_notify_functor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
