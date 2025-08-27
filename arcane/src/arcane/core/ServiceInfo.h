// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceInfo.h                                               (C) 2000-2025 */
/*                                                                           */
/* Informations d'un service.                                                */
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
//TODO: a supprimer. cela est conservé pour compatibilité avec le générateur des axl.
using Internal::ServiceInfo;
using Internal::ServiceAllInterfaceRegisterer;

namespace Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceInfoPrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations d'un service.
 */
class ARCANE_CORE_EXPORT ServiceInfo
: public IServiceInfo
{
 public:

  //! Constructeur
  ServiceInfo(const String& local_name,const VersionInfo& version,
              Integer valid_dimension);

  //! Destructeur
  ~ServiceInfo() override;

 public:

  //!@{ @name Méthodes d'accès héritées de IServiceInfo
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

  //!@{ @name Méthodes de construction spécifiques
  virtual void setAxlVersion(Real v) const;
  virtual void setCaseOptionsFileName(const String& fn);
  virtual void addFactory(IServiceFactory2* factory);
  virtual void setDefaultTagName(const String& value);
  virtual void setTagName(const String& value,const String& lang);
  virtual void setSingletonFactory(Internal::ISingletonServiceFactory* f);
  //!@}

  //! Infos sur les fabriques disponibles pour ce service
  IServiceFactoryInfo* factoryInfo() const override;
  void setFactoryInfo(IServiceFactoryInfo* sfi);

  void setAxlContent(const FileContent& file_content);

  int usageType() const override;

 public:
  
  // Fonction de création utilisée par les macros C++.
  static ServiceInfo* create(const ServiceProperty& sp,const char* filename,int lineno);
  
  // Fonction de création utilisée par le C#.
  // (le C# ne peut pas utiliser la méthode C++ à cause de ServiceProperty
  // qui contient un const char* qui sera collecté par le garbage collector)
  static ServiceInfo* create(const String& name,int service_type);

 private:

  ServiceInfoPrivate* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Internal
} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

