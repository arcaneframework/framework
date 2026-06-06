// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionsService.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Management of dataset options.                                            */
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
    // The service must be authorized for the dataset.
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
  // Iterates through the list of factories and tries to create a service with the desired name
  // that implements the correct interface.
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
    // In an else block: Symbol replacement does not apply to default values in .axl.
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
    // Uses the default value:
    // - if it was specified by the user, use this one.
    // - otherwise use the one associated with the default category.
    // - otherwise, the classic default value.
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
    // In an else block: Symbol replacement does not apply to default values in .axl.
    str_val = StringVariableReplace::replaceWithCmdLineArgs(params, str_val, true);
  }
  if (str_val.null() && !isOptional()){
    CaseOptionError::addOptionNotFoundError(doc,A_FUNCINFO,"@name",element);
    return;
  }
  m_service_name = str_val;

  // If the service can be null and the element is not present in the dataset,
  // it is considered that it should not be loaded.
  bool need_create = col->isPresent() || !isOptional();

  if (need_create){
    m_container->allocate(1);
    bool is_found = _tryCreateService(m_container,caseMng()->application(),str_val,0,this);

    if (!is_found && !m_allow_null){
      // The desired service was not found. This is an error.
      // Searches for the names of valid implementations to display in the corresponding error message.
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
  // !!! In XML, we start counting from 1, not 0.
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

  // We check if the user has provided an index too high for the option in the command line.
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

  // There will always be at least min_occurs options.
  // If there are not enough options in the dataset and in the command line parameters,
  // we add default services (if there is no default, there will be a crash).
  Integer final_size = std::max(size, std::max(min_occurs, max_in_param));

  ITraceMng* tm = traceMng();

  IApplication* app = caseMng()->application();
  ICaseDocumentFragment* doc = caseDocumentFragment();

  m_container->allocate(final_size);

  m_allocated_options.resize(final_size);
  m_services_name.resize(final_size);

  // First, we will have the dataset options: since we cannot define an index
  // for options in the dataset, they will necessarily be at the beginning and contiguous.
  // Then, if options are missing to reach min_occurs, we add default options.
  // If there is no default option, there will be an exception.
  // Finally, the user may have added options from the command line. We add them then.
  // If the user wants to modify dataset values from the command line, we replace the
  // options as we read them.
  for (Integer index = 0; index < final_size; ++index) {
    XmlNode element;

    String mesh_name;
    String str_val;

    // Command line parameters section.
    if (option_in_param.contains(index + 1)) {
      mesh_name = pco.getParameterOrNull(full_xpath, "@mesh-name", index + 1);
      str_val = pco.getParameterOrNull(full_xpath, "@name", index + 1);
    }
    // Dataset section.
    if (index < size && (mesh_name.null() || str_val.null())) {
      element = elem_list[index];
      if (!element.null()) {
        if (mesh_name.null())
          mesh_name = element.attrValue("mesh-name");
        if (str_val.null())
          str_val = element.attrValue("name");
      }
    }

    // Default value.
    if (mesh_name.null()) {
      mesh_name = meshName();
    }
    else {
      // In an else: Symbol replacement does not apply to .axl default values.
      mesh_name = StringVariableReplace::replaceWithCmdLineArgs(params, mesh_name, true);
    }

    // Default value.
    if (str_val.null()) {
      str_val = _defaultValue();
    }
    else {
      // In an else: Symbol replacement does not apply to .axl default values.
      str_val = StringVariableReplace::replaceWithCmdLineArgs(params, str_val, true);
    }

    // If we are not using the dataset options, we must create new options.
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

    // Now, this crash also concerns the case where there are no default values and not enough options to reach min_occurs.
    if (str_val.null())
      throw CaseOptionException("get_value", "@name");

    // TODO: check if we cannot create a CaseOptionService directly.
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
      // Search for valid implementation names
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
