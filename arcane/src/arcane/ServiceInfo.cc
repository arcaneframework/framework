// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceInfo.cc                                              (C) 2000-2019 */
/*                                                                           */
/* Informations d'un service.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/List.h"
#include "arcane/utils/VersionInfo.h"
#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/ServiceInfo.h"
#include "arcane/ServiceFactory.h"
#include "arcane/StringDictionary.h"
#include "arcane/IBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceInfoPrivate
{
 public:
  ServiceInfoPrivate(const String& local_name,const VersionInfo& version,
                     Integer valid_dimension);
  ~ServiceInfoPrivate();
 public:
  List<IServiceFactory2*> factories() { return m_factories; }
  void addFactory(IServiceFactory2* factory)
  {
    m_factories.add(factory);
    m_ref_factories.add(ReferenceCounter<IServiceFactory2>(factory));
  }
 public:
  String m_namespace_uri;
  String  m_local_name;
  VersionInfo m_version;
  Integer m_valid_dimension;
  StringList m_implemented_interfaces;
  String m_case_options_file_name;
 private:
  List<IServiceFactory2*> m_factories;
  UniqueArray<ReferenceCounter<IServiceFactory2>> m_ref_factories;
 public:
  ISingletonServiceFactory* m_singleton_factory;
  StringDictionary m_tag_names;
  String m_default_tag_name;
  Real m_axl_version;
  IServiceFactoryInfo* m_factory_info;
  FileContent m_axl_content;
  int m_usage_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceInfoPrivate::
ServiceInfoPrivate(const String& local_name,const VersionInfo& version,
                   Integer valid_dimension)
: m_namespace_uri(arcaneNamespaceURI())
, m_local_name(local_name)
, m_version(version)
, m_valid_dimension(valid_dimension)
, m_singleton_factory(nullptr)
, m_axl_version(0.0)
, m_factory_info(nullptr)
, m_usage_type(ST_None)
{
  m_default_tag_name = local_name.clone();
  m_default_tag_name = m_default_tag_name.lower();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceInfoPrivate::
~ServiceInfoPrivate()
{
  delete m_singleton_factory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceInfo::
ServiceInfo(const String& local_name,const VersionInfo& version,
            Integer valid_dimension)
: m_p(new ServiceInfoPrivate(local_name,version,valid_dimension))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceInfo::
~ServiceInfo()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ServiceInfo::
namespaceURI() const
{
  return m_p->m_namespace_uri;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ServiceInfo::
localName() const
{
  return m_p->m_local_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VersionInfo ServiceInfo::
version() const
{
  return m_p->m_version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ServiceInfo::
allowDimension(Integer n) const
{
  if (n==3 && (m_p->m_valid_dimension & Dim3))
    return true;
  if (n==2 && (m_p->m_valid_dimension & Dim2))
    return true;
  if (n==1 && (m_p->m_valid_dimension & Dim1))
    return true;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceInfo::
addImplementedInterface(const String& name)
{
  if (m_p->m_implemented_interfaces.contains(name))
    return;
  m_p->m_implemented_interfaces.add(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringCollection ServiceInfo::
implementedInterfaces() const
{
  return m_p->m_implemented_interfaces;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& ServiceInfo::
caseOptionsFileName() const
{
  return m_p->m_case_options_file_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceInfo::
setCaseOptionsFileName(const String& fn)
{
  m_p->m_case_options_file_name = fn;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceInfo::
addFactory(IServiceFactory2* factory)
{
  m_p->addFactory(factory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceFactory2Collection ServiceInfo::
factories() const
{
  return m_p->factories();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISingletonServiceFactory* ServiceInfo::
singletonFactory() const
{
  return m_p->m_singleton_factory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceInfo::
setSingletonFactory(ISingletonServiceFactory* f)
{
  delete m_p->m_singleton_factory;
  m_p->m_singleton_factory = f;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ServiceInfo::
tagName(const String& lang) const
{
  String v = m_p->m_tag_names.find(lang);
  if (v.null())
    v = m_p->m_default_tag_name;
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceInfo::
setDefaultTagName(const String& value)
{
  m_p->m_default_tag_name = value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceInfo::
setTagName(const String& value,const String& lang)
{
  m_p->m_tag_names.add(lang,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ServiceInfo::
axlVersion() const
{
  return m_p->m_axl_version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceInfo::
setAxlVersion(Real v) const
{
  m_p->m_axl_version = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IServiceFactoryInfo* ServiceInfo::
factoryInfo() const
{
  return m_p->m_factory_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceInfo::
setFactoryInfo(IServiceFactoryInfo* sfi)
{
  m_p->m_factory_info = sfi;
}

void ServiceInfo::
setAxlContent(const FileContent& file_content)
{
  m_p->m_axl_content = file_content;
}

const FileContent& ServiceInfo::
axlContent() const
{
  return m_p->m_axl_content;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ServiceInfo::
usageType() const
{
  return m_p->m_usage_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceInfo* ServiceInfo::
create(const ServiceProperty& sp,const char* filename,int lineno)
{
  ARCANE_UNUSED(filename);
  ARCANE_UNUSED(lineno);
  //TODO: utiliser les infos 'filename' et 'lineno'.

  // Attention à bien copier la chaîne issu de sp car il s'agit d'un const char*.
  String name = std::string_view(sp.name());
  ServiceInfo* si = new ServiceInfo(name,VersionInfo("0.0"),IServiceInfo::Dim2|IServiceInfo::Dim3);
  ServiceFactoryInfo* sfi = new ServiceFactoryInfo(si);
  si->setFactoryInfo(sfi);
  sfi->initProperties(sp.properties());
  si->m_p->m_usage_type = sp.type();
  return si;                                    
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceInfo* ServiceInfo::
create(const String& name,int service_type)
{
  ServiceProperty sp(name.localstr(),service_type);
  return create(sp,"none",0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

