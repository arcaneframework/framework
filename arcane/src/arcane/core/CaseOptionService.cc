// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionsService.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Gestion des options du jeu de données.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionService.h"

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/internal/ParameterCaseOption.h"

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
#include "arcane/core/internal/ICaseMngInternal.h"

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
  const ParameterListWithCaseOption& params = caseMng()->_internalImpl()->parameters();
  ICaseDocumentFragment* doc = caseDocumentFragment();

  const ParameterCaseOption pco{ params.getParameterCaseOption(doc->language()) };

  String mesh_name;
  {
    String reference_input = pco.getParameterOrNull(element.xpathFullName(), "@mesh-name", 1);
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
    String reference_input = pco.getParameterOrNull(element.xpathFullName(), "@name", 1);
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

  const ParameterListWithCaseOption& params = caseMng()->_internalImpl()->parameters();
  const ParameterCaseOption pco{ params.getParameterCaseOption(caseDocumentFragment()->language()) };

  XmlNode parent_element = configList()->parentElement();

  String full_xpath = String::format("{0}/{1}", parent_element.xpathFullName(), name());
  // !!! En XML, on commence par 1 et non 0.
  UniqueArray<Integer> option_in_param;
  pco.indexesInParam(full_xpath, option_in_param, true);

  Integer size = elem_list.size();

  bool is_optional = configList()->isOptional();

  if (size == 0 && option_in_param.empty() && is_optional) {
    return;
  }

  Integer min_occurs = configList()->minOccurs();
  Integer max_occurs = configList()->maxOccurs();

  Integer max_in_param = 0;

  // On regarde si l'utilisateur n'a pas mis un indice trop élevé pour l'option dans la ligne de commande.
  if (!option_in_param.empty()) {
    max_in_param = option_in_param[0];
    for (Integer index : option_in_param) {
      if (index > max_in_param)
        max_in_param = index;
    }
    if (max_occurs >= 0) {
      if (max_in_param > max_occurs) {
        StringBuilder msg = "Bad number of occurences in command line (greater than max)";
        msg += " index_max_in_param=";
        msg += max_in_param;
        msg += " max_occur=";
        msg += max_occurs;
        msg += " option=";
        msg += full_xpath;
        throw CaseOptionException(A_FUNCINFO, msg.toString(), true);
      }
    }
  }

  if (max_occurs >= 0) {
    if (size > max_occurs) {
      StringBuilder msg = "Bad number of occurences (greater than max)";
      msg += " nb_occur=";
      msg += size;
      msg += " max_occur=";
      msg += max_occurs;
      msg += " option=";
      msg += full_xpath;
      throw CaseOptionException(A_FUNCINFO, msg.toString(), true);
    }
  }

  // Il y aura toujours au moins min_occurs options.
  // S'il n'y a pas assez l'options dans le jeu de données et dans les paramètres de la
  // ligne de commande, on ajoute des services par défaut (si pas de défaut, il y aura un plantage).
  Integer final_size = std::max(size, std::max(min_occurs, max_in_param));

  ITraceMng* tm = traceMng();

  IApplication* app = caseMng()->application();
  ICaseDocumentFragment* doc = caseDocumentFragment();

  m_container->allocate(final_size);

  m_allocated_options.resize(final_size);
  m_services_name.resize(final_size);

  // D'abord, on aura les options du jeu de données : comme on ne peut pas définir un indice
  // pour les options dans le jeu de données, elles seront forcément au début et seront contigües.
  // Puis, s'il manque des options pour atteindre le min_occurs, on ajoute des options par défaut.
  // S'il n'y a pas d'option par défaut, il y aura une exception.
  // Enfin, l'utilisateur peut avoir ajouté des options à partir de la ligne de commande. On les ajoute alors.
  // Si l'utilisateur souhaite modifier des valeurs du jeu de données à partir de la ligne de commande, on
  // remplace les options au fur et à mesure de la lecture.
  for (Integer index = 0; index < final_size; ++index) {
    XmlNode element;

    String mesh_name;
    String str_val;

    // Partie paramètres de la ligne de commande.
    if (option_in_param.contains(index + 1)) {
      mesh_name = pco.getParameterOrNull(full_xpath, "@mesh-name", index + 1);
      str_val = pco.getParameterOrNull(full_xpath, "@name", index + 1);
    }
    // Partie jeu de données.
    if (index < size && (mesh_name.null() || str_val.null())) {
      element = elem_list[index];
      if (!element.null()) {
        if (mesh_name.null())
          mesh_name = element.attrValue("mesh-name");
        if (str_val.null())
          str_val = element.attrValue("name");
      }
    }

    // Valeur par défaut.
    if (mesh_name.null()) {
      mesh_name = meshName();
    }
    else {
      // Dans un else : Le remplacement de symboles ne s'applique pas pour les valeurs par défault du .axl.
      mesh_name = StringVariableReplace::replaceWithCmdLineArgs(params, mesh_name, true);
    }

    // Valeur par défaut.
    if (str_val.null()) {
      str_val = _defaultValue();
    }
    else {
      // Dans un else : Le remplacement de symboles ne s'applique pas pour les valeurs par défault du .axl.
      str_val = StringVariableReplace::replaceWithCmdLineArgs(params, str_val, true);
    }

    // Si l'on n'utilise pas les options du jeu de données, on doit créer de nouvelles options.
    if (element.null()) {
      element = parent_element.createElement(name());

      element.setAttrValue("mesh-name", mesh_name);
      element.setAttrValue("name", str_val);
    }

    tm->info(5) << "CaseOptionMultiServiceImpl name=" << name()
                << " index=" << index
                << " v=" << str_val
                << " default_value='" << _defaultValue() << "'"
                << " mesh=" << meshHandle().meshName();

    // Maintenant, ce plantage concerne aussi le cas où il n'y a pas de valeurs par défaut et qu'il n'y a
    // pas assez d'options pour atteindre le min_occurs.
    if (str_val.null())
      throw CaseOptionException("get_value", "@name");

    // TODO: regarder si on ne peut pas créer directement un CaseOptionService.
    auto* coptions = new CaseOptions(configList(), name(), parent_element, false, true);
    if (coptions->_setMeshHandleAndCheckDisabled(mesh_name)) {
      delete coptions;
      continue;
    }
    coptions->configList()->_internalApi()->setRootElement(element);
    bool is_found = _tryCreateService(m_container, app, str_val, index, coptions);

    if (!is_found) {
      tm->info(5) << "CaseOptionMultiServiceImpl name=" << name()
                  << " index=" << index
                  << " service not found";
      delete coptions;
      coptions = nullptr;
      // Recherche les noms des implémentations valides
      StringUniqueArray valid_names;
      getAvailableNames(valid_names);
      CaseOptionError::addError(doc, A_FUNCINFO, element.xpathFullName(),
                                String::format("Unable to find a service named '{0}' (valid values:{1})",
                                               str_val, valid_names),
                                true);
    }
    m_services_name[index] = str_val;
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

