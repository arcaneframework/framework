// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceInfo.h                                               (C) 2000-2025 */
/*                                                                           */
/* Information about a service.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SERVICEINFO_H
#define ARCANE_CORE_SERVICEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/FileContent.h"
#include "arcane/core/IServiceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
//TODO: to be removed. This is kept for compatibility with the axl generator.
using Internal::ServiceAllInterfaceRegisterer;
using Internal::ServiceInfo;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceInfoPrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Information about a service.
 */
class ARCANE_CORE_EXPORT ServiceInfo
: public IServiceInfo
{
 public:

  //! Constructor
  ServiceInfo(const String& local_name, const VersionInfo& version,
              Integer valid_dimension);

  //! Destructor
  ~ServiceInfo() override;

 public:

  //!@{ @name Methods inherited from IServiceInfo
  String localName() const override;
  String namespaceURI() const override;
  VersionInfo version() const override;
  Real axlVersion() const override;
  bool allowDimension(Integer n) const override;
  void addImplementedInterface(const String& name) override;
  StringCollection implementedInterfaces() const override;
  const String& caseOptionsFileName() const override;
  ServiceFactory2Collection factories() const override;
  ISingletonServiceFactory* singletonFactory() const override;
  String tagName(const String& lang) const override;
  const FileContent& axlContent() const override;
  //!@}

  //!@{ @name Specific construction methods
  virtual void setAxlVersion(Real v) const;
  virtual void setCaseOptionsFileName(const String& fn);
  virtual void addFactory(IServiceFactory2* factory);
  virtual void setDefaultTagName(const String& value);
  virtual void setTagName(const String& value, const String& lang);
  virtual void setSingletonFactory(Internal::ISingletonServiceFactory* f);
  //!@}

  //! Info on the factories available for this service
  IServiceFactoryInfo* factoryInfo() const override;
  void setFactoryInfo(IServiceFactoryInfo* sfi);

  void setAxlContent(const FileContent& file_content);

  int usageType() const override;

 public:

  // Creation function used by C++ macros.
  static ServiceInfo* create(const ServiceProperty& sp, const char* filename, int lineno);

  // Creation function used by C#.
  // (C# cannot use the C++ method because of ServiceProperty
  // which contains a const char* that will be collected by the garbage collector)
  static ServiceInfo* create(const String& name, int service_type);

 private:

  ServiceInfoPrivate* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
