// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptions.cc                                              (C) 2000-2023 */
/*                                                                           */
/* Gestion des options du jeu de données.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionService.h"

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/IServiceFactory.h"
#include "arcane/core/CaseOptionBuildInfo.h"
#include "arcane/core/CaseOptionException.h"
#include "arcane/core/CaseOptionError.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/ICaseDocumentVisitor.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/internal/ICaseOptionListInternal.h"
#include "arcane/core/internal/StringVariableReplace.h"

#include <typeinfo>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace {

void
_getAvailableServiceNames(ICaseOptionServiceContainer* container,IApplication* app,
                          StringArray& names)
{
  for( ServiceFactory2Collection::Enumerator i(app->serviceFactories2()); ++i; ){
    Internal::IServiceFactory2* sf2 = *i;
    IServiceInfo* si = sf2->serviceInfo();
    // Il faut que le service soit autorisé pour le jeu de données.
    if (!(si->usageType() & Arcane::ST_CaseOption))
      continue;
    if (container->hasInterfaceImplemented(sf2)){
      names.add(sf2->serviceInfo()->localName());
    }
  }
}

bool
_tryCreateService(ICaseOptionServiceContainer* container,IApplication* app,
                  const String& service_name,Integer index,ICaseOptions* opt)
{
  // Parcours la liste des fabriques et essaie de créer un service avec le nom
  // qu'on souhaite et qui implémente la bonne interface.
  bool is_found = false;
  ServiceBuildInfoBase sbi(_arcaneDeprecatedGetSubDomain(opt),opt);
  for( ServiceFactory2Collection::Enumerator i(app->serviceFactories2()); ++i; ){
    Internal::IServiceFactory2* sf2 = *i;
    IServiceInfo* si = sf2->serviceInfo();
    if (si->localName()==service_name && container->tryCreateService(index,sf2,sbi)){
      opt->setCaseServiceInfo(si);
      is_found = true;
      break;
    }
  }
  return is_found;
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionService::
setMeshName(const String& mesh_name)
{
  m_impl->setMeshName(mesh_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionService::
meshName() const
{
  return m_impl->meshName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiService::
setMeshName(const String& mesh_name)
{
  m_impl->setMeshName(mesh_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionMultiService::
meshName() const
{
  return m_impl->meshName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionServiceImpl::
CaseOptionServiceImpl(const CaseOptionBuildInfo& cob,bool allow_null,bool is_optional)
: CaseOptions(cob.caseOptionList(),cob.name())
, m_name(cob.name())
, m_default_value(cob.defaultValue())
, m_element(cob.element())
, m_allow_null(allow_null)
, m_is_optional(is_optional)
, m_is_override_default(false)
, m_container(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionServiceImpl::
print(const String& lang,std::ostream& o) const
{
  ARCANE_UNUSED(lang);
  o << serviceName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionServiceImpl::
visit(ICaseDocumentVisitor* visitor) const
{
  visitor->beginVisit(this);
  CaseOptions::visit(visitor);
  visitor->endVisit(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionServiceImpl::
setContainer(ICaseOptionServiceContainer* container)
{
  m_container = container;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionServiceImpl::
_readPhase1()
{
  if (!m_container)
    ARCANE_FATAL("null 'm_container'. did you called setContainer() method ?");

  ITraceMng* tm = traceMng();

  _setTranslatedName();
  ICaseOptionList* col = configList();

  XmlNode child = m_element.child(rootTagName());  

  if (child.null()) {
    col->_internalApi()->setRootElementWithParent(m_element);
  }
  else {
    if (col->rootElement() != child) // skip when rootElement already set to child (may appear in subDomain service)
      col->_internalApi()->setRootElement(child);
  }

  XmlNode element = col->rootElement();
  const ParameterList& params = caseMng()->application()->applicationInfo().commandLineArguments().parameters();
  ICaseDocumentFragment* doc = caseDocumentFragment();

  String mesh_name;
  {
    String path = element.xpathFullName() + "/@mesh-name";

    // On retire le "//case/" ou le "//cas/" du début.
    StringView sv;
    if (doc->language() == "fr")
      sv = path.view().subView(6);
    else
      sv = path.view().subView(7);

    String reference_input = params.getParameterOrNull(sv);
    if (!reference_input.null())
      mesh_name = reference_input;
    else
      mesh_name = element.attrValue("mesh-name");
  }

  if (mesh_name.null()) {
    mesh_name = meshName();
  }
  else {
    // Dans un else : Le remplacement de symboles ne s'applique pas pour les valeurs par défault du .axl.
    mesh_name = StringVariableReplace::replaceWithCmdLineArgs(params, mesh_name, true);
  }

  tm->info(5) << "** CaseOptionService::read() ELEMENT <" << rootTagName() << "> " << col->rootElement().name()
              << " full=" << col->rootElement().xpathFullName()
              << " is_present=" << col->isPresent()
              << " is_optional=" << isOptional()
              << " allow_null=" << m_allow_null
              << " mesh-name=" << mesh_name
              << "\n";


  if (_setMeshHandleAndCheckDisabled(mesh_name))
    return;

  String str_val;
  {
    String path = element.xpathFullName() + "/@name";

    // On retire le "//case/" ou le "//cas/" du début.
    StringView sv;
    if (doc->language() == "fr")
      sv = path.view().subView(6);
    else
      sv = path.view().subView(7);

    String reference_input = params.getParameterOrNull(sv);
    if (!reference_input.null())
      str_val = reference_input;
    else
      str_val = element.attrValue("name");
  }

  //cerr << "** STR_VAL <" << str_val << " - " << m_default_value << ">\n";

  if (str_val.null()){
    // Utilise la valeur par défaut :
    // - si elle a été spécifiée par l'utilisateur, utilise celle-ci.
    // - sinon utilise celle de la catégorie associée aux défauts.
    // - sinon, la valeur par défaut classique.
    if (!m_is_override_default){
      String category = doc->defaultCategory();
      if (!category.null()){
        String v = m_default_values.find(category);
        if (!v.null()){
          m_default_value = v;
        }
      }
    }
    str_val = m_default_value;
  }
  else {
    // Dans un else : Le remplacement de symboles ne s'applique pas pour les valeurs par défault du .axl.
    str_val = StringVariableReplace::replaceWithCmdLineArgs(params, str_val, true);
  }
  if (str_val.null() && !isOptional()){
    CaseOptionError::addOptionNotFoundError(doc,A_FUNCINFO,"@name",element);
    return;
  }
  m_service_name = str_val;

  // Si le service peut-être nul et que l'élément n'est pas présent dans le jeu de données,
  // on considère qu'il ne doit pas être chargé.
  bool need_create = col->isPresent() || !isOptional();

  if (need_create){
    m_container->allocate(1);
    bool is_found = _tryCreateService(m_container,caseMng()->application(),str_val,0,this);

    if (!is_found && !m_allow_null){
      // Le service souhaité n'est pas trouvé. Il s'agit d'une erreur.
      // Recherche les noms des implémentations valides pour affichage dans le message
      // d'erreur correspondant.
      StringUniqueArray valid_names;
      getAvailableNames(valid_names);
      CaseOptionError::addError(doc,A_FUNCINFO,element.xpathFullName(),
                                String::format("Unable to find a service named '{0}' (valid values:{1})",
                                               str_val,valid_names),true);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionServiceImpl::
read(eCaseOptionReadPhase read_phase)
{
  if (read_phase==eCaseOptionReadPhase::Phase1)
    _readPhase1();
  CaseOptions::read(read_phase);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionServiceImpl::
setDefaultValue(const String& def_value)
{
  if (!m_service_name.null()){
    String xpath_name = configList()->rootElement().xpathFullName();
    ARCANE_FATAL("Can not set default service name because service is already allocated (option='{0}')",
                 xpath_name);
  }
  m_default_value = def_value;
  m_is_override_default = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionServiceImpl::
addDefaultValue(const String& category,const String& value)
{
  m_default_values.add(category,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionServiceImpl::
getAvailableNames(StringArray& names) const
{
  _getAvailableServiceNames(m_container,caseMng()->application(),names);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionMultiServiceImpl::
CaseOptionMultiServiceImpl(const CaseOptionBuildInfo& cob,bool allow_null)
: CaseOptionsMulti(cob.caseOptionList(),cob.name(),cob.element(),cob.minOccurs(),cob.maxOccurs())
, m_allow_null(allow_null)
, m_default_value(cob.defaultValue())
, m_notify_functor(nullptr)
, m_container(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionMultiServiceImpl::
~CaseOptionMultiServiceImpl()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiServiceImpl::
setContainer(ICaseOptionServiceContainer* container)
{
  m_container = container;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiServiceImpl::
visit(ICaseDocumentVisitor* visitor) const
{
  Integer index = 0;
  for( const auto& o : m_allocated_options ){
    visitor->beginVisit(this,index);
    o->visit(visitor);
    visitor->endVisit(this,index);
    ++index;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiServiceImpl::
multiAllocate(const XmlNodeList& elem_list)
{
  if (!m_container)
    ARCANE_FATAL("null 'm_container'. did you called setContainer() method ?");

  Integer size = elem_list.size();

  if (size==0)
    return;

  m_services_name.resize(size);

  ITraceMng* tm = traceMng();

  m_container->allocate(size);
  m_allocated_options.resize(size);
  IApplication* app = caseMng()->application();
  XmlNode parent_element = configList()->parentElement();
  ICaseDocumentFragment* doc = caseDocumentFragment();

  String mesh_name = meshName();
  if (_setMeshHandleAndCheckDisabled(mesh_name))
    return;

  for( Integer index=0; index<size; ++index ){
    XmlNode element = elem_list[index];
      
    String str_val = element.attrValue("name");
    m_services_name[index] = str_val;
    tm->info(5) << "CaseOptionMultiServiceImpl name=" << name()
                << " index=" << index
                << " v=" << str_val
                << " default_value='" << _defaultValue() << "'"
                << " mesh=" << meshHandle().meshName();
        
    if (str_val.null())
      str_val = _defaultValue();
    if (str_val.null())
      throw CaseOptionException("get_value","@name",element);
    // TODO: regarder si on ne peut pas créer directement un CaseOptionService.
    CaseOptions* coptions = new CaseOptions(configList(),name(),parent_element,false,true);
    coptions->configList()->_internalApi()->setRootElement(element);
    bool is_found = _tryCreateService(m_container,app,str_val,index,coptions);

    if (!is_found){
      tm->info(5) << "CaseOptionMultiServiceImpl name=" << name()
                  << " index=" << index
                  << " service not found";
      delete coptions;
      coptions = nullptr;
      // Recherche les noms des implémentations valides
      StringUniqueArray valid_names;
      getAvailableNames(valid_names);
      CaseOptionError::addError(doc,A_FUNCINFO,element.xpathFullName(),
                                String::format("Unable to find a service named '{0}' (valid values:{1})",
                                               str_val,valid_names),true);
    }
    m_allocated_options[index] = coptions;
  }
  if (m_notify_functor)
    m_notify_functor->executeFunctor();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiServiceImpl::
getAvailableNames(StringArray& names) const
{
  _getAvailableServiceNames(m_container,caseMng()->application(),names);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

