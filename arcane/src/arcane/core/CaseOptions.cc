// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptions.cc                                              (C) 2000-2023 */
/*                                                                           */
/* Data set options management.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptions.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/ICaseFunction.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/StringDictionary.h"
#include "arcane/core/CaseNodeNames.h"
#include "arcane/core/CaseOptionError.h"
#include "arcane/core/ICaseDocumentVisitor.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/internal/ICaseMngInternal.h"
#include "arcane/core/internal/ICaseOptionListInternal.h"

#include "arcane/core/CaseOptionsMulti.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace AxlOptionsBuilder
{
  extern "C++" IXmlDocumentHolder*
  documentToXml(const Document& d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
class CaseOptionsPrivate
{
 public:

  CaseOptionsPrivate(ICaseMng* cm, const String& name)
  : m_case_mng(cm)
  , m_name(name)
  , m_true_name(name)
  , m_mesh_handle(cm->meshMng()->defaultMeshHandle())
  {
  }

  CaseOptionsPrivate(ICaseOptionList* co_list, const String& name)
  : m_case_mng(co_list->caseMng())
  , m_name(name)
  , m_true_name(name)
  , m_mesh_handle(co_list->meshHandle())
  {
    if (m_mesh_handle.isNull())
      m_mesh_handle = m_case_mng->meshMng()->defaultMeshHandle();
  }

  ~CaseOptionsPrivate()
  {
    delete m_own_case_document_fragment;
  }

 public:

  ICaseOptionList* m_parent = nullptr;
  ICaseMng* m_case_mng;
  ReferenceCounter<ICaseOptionList> m_config_list;
  IModule* m_module = nullptr; //!< Associated module or 0 if none.
  IServiceInfo* m_service_info = nullptr; //!< Associated service or 0 if none.
  String m_name;
  String m_true_name;
  bool m_is_multi = false;
  bool m_is_translated_name_set = false;
  bool m_is_phase1_read = false;
  StringDictionary m_name_translations;
  ICaseFunction* m_activate_function = nullptr; //!< Function indicating activation status
  bool m_is_case_mng_registered = false;
  MeshHandle m_mesh_handle;
  // non-null if we own our own document instance
  ICaseDocumentFragment* m_own_case_document_fragment = nullptr;
  Ref<ICaseMng> m_case_mng_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseMng* cm, const String& name)
: m_p(new CaseOptionsPrivate(cm, name))
{
  m_p->m_config_list = ICaseOptionListInternal::create(cm, this, XmlNode());
  m_p->m_is_case_mng_registered = true;
  cm->registerOptions(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseOptionList* parent, const String& aname)
: m_p(new CaseOptionsPrivate(parent, aname))
{
  m_p->m_config_list = ICaseOptionListInternal::create(parent, this, XmlNode());
  _setParent(parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseMng* cm, const String& aname, const XmlNode& parent_elem)
: m_p(new CaseOptionsPrivate(cm, aname))
{
  m_p->m_config_list = ICaseOptionListInternal::create(cm, this, parent_elem);
  m_p->m_is_case_mng_registered = true;
  cm->registerOptions(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseOptionList* parent, const String& aname,
            const XmlNode& parent_elem, bool is_optional, bool is_multi)
: m_p(new CaseOptionsPrivate(parent, aname))
{
  ICaseOptionList* col = ICaseOptionListInternal::create(parent, this, parent_elem, is_optional, is_multi);
  m_p->m_config_list = col;
  _setParent(parent);
  if (is_multi)
    _setTranslatedName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseMng* cm, const String& aname, ICaseOptionList* config_list)
: m_p(new CaseOptionsPrivate(cm, aname))
{
  m_p->m_config_list = config_list;
  m_p->m_is_case_mng_registered = true;
  cm->registerOptions(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseOptionList* parent, const String& aname,
            ICaseOptionList* config_list)
: m_p(new CaseOptionsPrivate(parent->caseMng(), aname))
{
  m_p->m_config_list = config_list;
  _setParent(parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseMng* cm, const XmlContent& xml_content)
: m_p(new CaseOptionsPrivate(cm, "dynamic-options"))
{
  // This constructor is for dynamically created options
  IXmlDocumentHolder* xml_doc = xml_content.m_document;
  XmlNode parent_elem = xml_doc->documentNode().documentElement();
  m_p->m_config_list = ICaseOptionListInternal::create(cm, this, parent_elem);
  m_p->m_own_case_document_fragment = cm->_internalImpl()->createDocumentFragment(xml_doc);
  // Keeps a reference to the ICaseMng in case this option
  // is destroyed after the end of the calculation and the destruction of subdomains.
  m_p->m_case_mng_ref = cm->toReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
~CaseOptions()
{
  detach();
  if (m_p->m_is_case_mng_registered)
    m_p->m_case_mng->unregisterOptions(this);
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
detach()
{
  if (m_p->m_parent)
    m_p->m_parent->removeChild(this);
  m_p->m_parent = nullptr;
  m_p->m_config_list = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptions::
rootTagTrueName() const
{
  return m_p->m_true_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptions::
rootTagName() const
{
  return m_p->m_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseOptions::
isPresent() const
{
  return m_p->m_config_list->isPresent();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptions::
xpathFullName() const
{
  return m_p->m_config_list->xpathFullName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
addAlternativeNodeName(const String& lang, const String& name)
{
  // Translations should not be modified once the translated name
  // has been set. This can happen with services if they have a translation
  // in their axl. In this case, this last one overrides the parent option,
  // which can make the names inconsistent.
  if (m_p->m_is_translated_name_set)
    return;
  m_p->m_name_translations.add(lang, name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseOptionList* CaseOptions::
configList()
{
  return m_p->m_config_list.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ICaseOptionList* CaseOptions::
configList() const
{
  return m_p->m_config_list.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IServiceInfo* CaseOptions::
caseServiceInfo() const
{
  return m_p->m_service_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IModule* CaseOptions::
caseModule() const
{
  return m_p->m_module;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
setCaseServiceInfo(IServiceInfo* m)
{
  m_p->m_service_info = m;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
setCaseModule(IModule* m)
{
  m_p->m_module = m;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseMng* CaseOptions::
caseMng() const
{
  return m_p->m_case_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* CaseOptions::
traceMng() const
{
  return m_p->m_case_mng->traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* CaseOptions::
subDomain() const
{
  return m_p->m_case_mng->subDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle CaseOptions::
meshHandle() const
{
  return m_p->m_mesh_handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* CaseOptions::
mesh() const
{
  return meshHandle().mesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseDocument* CaseOptions::
caseDocument() const
{
  return caseMng()->caseDocument();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseDocumentFragment* CaseOptions::
caseDocumentFragment() const
{
  auto* x = m_p->m_own_case_document_fragment;
  if (x)
    return x;
  return caseMng()->caseDocumentFragment();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
_setMeshHandle(const MeshHandle& handle)
{
  traceMng()->info(5) << "SetMeshHandle for " << m_p->m_name << " mesh_name=" << handle.meshName();
  m_p->m_mesh_handle = handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
_setParent(ICaseOptionList* parent)
{
  parent->addChild(this);
  m_p->m_parent = parent;
  _setMeshHandle(parent->meshHandle());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Positions the mesh associated with this option.
 *
 * If \a mesh_name is null or empty, the mesh associated with this
 * option is that of the parent option. If the option has no parent, it is the default mesh.
 *
 * If \a mesh_name is not null, there are two possibilities:
 * - if the specified mesh exists, the option will be associated with that mesh
 * - if it does not exist, the option is disabled and any potential child options
 * will not be read. This latter case occurs, for example, if a service
 * is associated with an additional mesh but that mesh is optional.
 * In this case, the option must not be read.
 *
 * \retval true if the option is disabled following this call.
 */
bool CaseOptions::
_setMeshHandleAndCheckDisabled(const String& mesh_name)
{
  if (mesh_name.empty()) {
    // My mesh is that of my parent
    if (m_p->m_parent)
      _setMeshHandle(m_p->m_parent->meshHandle());
  }
  else {
    // A mesh different from the default mesh is associated with the option.
    // Retrieve the associated MeshHandle if it exists. If it doesn't,
    // disable the option.
    // If no mesh with the name we are looking for exists, do not allocate the service
    MeshHandle* handle = caseMng()->meshMng()->findMeshHandle(mesh_name, false);
    if (!handle) {
      m_p->m_config_list->disable();
      return true;
    }
    _setMeshHandle(*handle);
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
_setTranslatedName()
{
  String lang = caseDocumentFragment()->language();
  if (m_p->m_is_translated_name_set)
    traceMng()->info() << "WARNING: translated name already set for " << m_p->m_name;
  if (lang.null())
    m_p->m_name = m_p->m_true_name;
  else {
    String tr = m_p->m_name_translations.find(lang);
    if (!tr.null()) {
      m_p->m_name = tr;
    }
  }
  m_p->m_is_translated_name_set = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptions::
translatedName(const String& lang) const
{
  if (!lang.null()) {
    String tr = m_p->m_name_translations.find(lang);
    if (!tr.null())
      return tr;
  }
  return m_p->m_true_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseFunction* CaseOptions::
activateFunction()
{
  return m_p->m_activate_function;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
read(eCaseOptionReadPhase read_phase)
{
  ITraceMng* tm = traceMng();
  bool is_phase1 = read_phase == eCaseOptionReadPhase::Phase1;
  if (is_phase1 && m_p->m_is_phase1_read)
    return;

  if (is_phase1 && !m_p->m_is_translated_name_set)
    _setTranslatedName();

  m_p->m_config_list->readChildren(is_phase1);

  if (is_phase1) {
    ICaseDocumentFragment* doc = caseDocumentFragment();
    // Read the activation function (if present)
    XmlNode velem = m_p->m_config_list->rootElement();
    CaseNodeNames* cnn = doc->caseNodeNames();
    String func_activation_name = velem.attrValue(cnn->function_activation_ref);
    if (!func_activation_name.null()) {
      ICaseFunction* func = caseMng()->findFunction(func_activation_name);
      if (!func) {
        CaseOptionError::addError(doc, A_FUNCINFO, velem.xpathFullName(),
                                  String::format("No function with the name '{0}' exists",
                                                 func_activation_name));
      }
      else if (func->paramType() != ICaseFunction::ParamReal) {
        CaseOptionError::addError(doc, A_FUNCINFO, velem.xpathFullName(),
                                  String::format("The function '{0}' requires a parameter of type 'time'",
                                                 func_activation_name));
      }
      else if (func->valueType() != ICaseFunction::ValueBool) {
        CaseOptionError::addError(doc, A_FUNCINFO, velem.xpathFullName(),
                                  String::format("The function '{0}' requires a parameter of type 'bool'",
                                                 func_activation_name));
      }
      else {
        m_p->m_activate_function = func;
        tm->info() << "Use the function '" << func->name() << "' to activate the option "
                   << velem.xpathFullName();
      }
    }
    // Check that the 'function' element is not present
    {
      String func_name = velem.attrValue(cnn->function_ref);
      if (!func_name.null())
        CaseOptionError::addError(doc, A_FUNCINFO, velem.xpathFullName(),
                                  String::format("Attribute <{0}> invalid.",
                                                 cnn->function_ref));
    }
    m_p->m_is_phase1_read = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Adds unrecognized elements to \a nlist.
 */
void CaseOptions::
addInvalidChildren(XmlNodeList& nlist)
{
  m_p->m_config_list->_internalApi()->addInvalidChildren(nlist);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
printChildren(const String& lang, int indent)
{
  m_p->m_config_list->printChildren(lang, indent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
visit(ICaseDocumentVisitor* visitor) const
{
  m_p->m_config_list->visit(visitor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
deepGetChildren(Array<CaseOptionBase*>& col)
{
  m_p->m_config_list->deepGetChildren(col);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ReferenceCounter<ICaseOptions> CaseOptions::
createDynamic(ICaseMng* cm, const AxlOptionsBuilder::Document& options_doc)
{
  XmlContent content;

  IXmlDocumentHolder* xml_doc = AxlOptionsBuilder::documentToXml(options_doc);
  content.m_document = xml_doc;

  auto* opt = new CaseOptions(cm, content);
  return ReferenceCounter<ICaseOptions>(opt);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ICaseOptions> CaseOptions::
toReference()
{
  return makeRef<ICaseOptions>(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionsMulti::
CaseOptionsMulti(ICaseMng* cm, const String& aname, const XmlNode& parent_element,
                 Integer min_occurs, Integer max_occurs)
: CaseOptions(cm, aname,
              ICaseOptionListInternal::create(this, this, cm, parent_element,
                                              min_occurs, max_occurs))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionsMulti::
CaseOptionsMulti(ICaseOptionList* parent, const String& aname,
                 const XmlNode& parent_element,
                 Integer min_occurs, Integer max_occurs)
: CaseOptions(parent, aname,
              ICaseOptionListInternal::create(this, this, parent,
                                              parent_element, min_occurs, max_occurs))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT ISubDomain*
_arcaneDeprecatedGetSubDomain(ICaseOptions* opt)
{
  return opt->subDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
