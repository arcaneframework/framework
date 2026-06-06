// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionServiceImpl.h                                     (C) 2000-2025 */
/*                                                                           */
/* Implementation of a dataset option using a service.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEOPTIONSERVICEIMPL_H
#define ARCANE_CORE_CASEOPTIONSERVICEIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Functor.h"

#include "arcane/core/CaseOptions.h"
#include "arcane/core/CaseOptionsMulti.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IService;
class CaseOptionBuildInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a service instance container.
 * \todo: add reference counter
 */
class ARCANE_CORE_EXPORT ICaseOptionServiceContainer
{
 public:
  virtual ~ICaseOptionServiceContainer() = default;
 public:
  virtual bool tryCreateService(Integer index,Internal::IServiceFactory2* factory,const ServiceBuildInfoBase& sbi) =0;
  virtual bool hasInterfaceImplemented(Internal::IServiceFactory2*) const =0;
  //! Allocates an array for \a size elements
  virtual void allocate(Integer size) =0;
  //! Returns the number of elements in the array.
  virtual Integer nbElem() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \ingroup CaseOption
 * \brief Base class for implementing options using services.
 *
 * This class is internal to Arcane. The class to use is 'CaseOptionService'.
 */
class ARCANE_CORE_EXPORT CaseOptionServiceImpl
: public CaseOptions
{
 public:

  CaseOptionServiceImpl(const CaseOptionBuildInfo& cob,bool allow_null,bool is_optional);

 public:

  void read(eCaseOptionReadPhase phase) override;
  String serviceName() const { return m_service_name; }
  bool isOptional() const { return m_is_optional; }

  //! Returns the valid implementation names for this service in \a names
  virtual void getAvailableNames(StringArray& names) const;
  void visit(ICaseDocumentVisitor* visitor) const override;

  void setDefaultValue(const String& def_value);
  void addDefaultValue(const String& category,const String& value);

  /*!
   * \brief Positions the instance container.
   *
   * \a container remains the property of the caller, who must manage
   * its lifetime.
   */
  void setContainer(ICaseOptionServiceContainer* container);

  void setMeshName(const String& mesh_name) { m_mesh_name = mesh_name; }
  String meshName() const { return m_mesh_name; }

 protected:

  virtual void print(const String& lang,std::ostream& o) const;

 protected:

  String _defaultValue() const { return m_default_value; }

 private:

  String m_name;
  String m_default_value;
  String m_service_name;
  String m_mesh_name;
  XmlNode m_element; //!< Option element
  bool m_allow_null;
  bool m_is_optional;
  bool m_is_override_default;
  //! List of default values by category.
  StringDictionary m_default_values;
  ICaseOptionServiceContainer* m_container;

 private:

  void _readPhase1();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for a service option that can appear multiple times.
 *
 * You must call setContainer() to position a container
 * before using this class.
 */
class ARCANE_CORE_EXPORT CaseOptionMultiServiceImpl
: public CaseOptionsMulti
{
 public:

  CaseOptionMultiServiceImpl(const CaseOptionBuildInfo& cob,bool allow_null);
  ~CaseOptionMultiServiceImpl();

 public:

  //! Returns the valid implementation names for this service in \a names
  void getAvailableNames(StringArray& names) const;
  //! Name of the nth service
  String serviceName(Integer index) const
  {
    return m_services_name[index];
  }

  void multiAllocate(const XmlNodeList&) override;
  void visit(ICaseDocumentVisitor* visitor) const override;

  /*!
   * \brief Positions the instance container.
   *
   * \a container remains the property of the caller, who must manage
   * its lifetime.
   */
  void setContainer(ICaseOptionServiceContainer* container);

  void setMeshName(const String& mesh_name) { m_mesh_name = mesh_name; }
  String meshName() const { return m_mesh_name; }

 public:

  void _setNotifyAllocateFunctor(IFunctor* f)
  {
    m_notify_functor = f;
  }

 protected:

  String _defaultValue() const { return m_default_value; }

 protected:

  bool m_allow_null;
  String m_default_value;
  String m_mesh_name;
  IFunctor* m_notify_functor;
  ICaseOptionServiceContainer* m_container;
  //! List of allocated options that must be deleted.
  UniqueArray<ReferenceCounter<ICaseOptions>> m_allocated_options;
  //! Service names for each occurrence
  UniqueArray<String> m_services_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
