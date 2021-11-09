// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptions.cc                                              (C) 2000-2019 */
/*                                                                           */
/* Gestion des options du jeu de données.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/CaseOptionException.h"
#include "arcane/CaseOptions.h"
#include "arcane/CaseOptionBuildInfo.h"
#include "arcane/XmlNode.h"
#include "arcane/XmlNodeList.h"
#include "arcane/XmlNodeIterator.h"
#include "arcane/ICaseFunction.h"
#include "arcane/ICaseMng.h"
#include "arcane/ICaseDocument.h"
#include "arcane/ArcaneException.h"
#include "arcane/ISubDomain.h"
#include "arcane/MathUtils.h"
#include "arcane/StringDictionary.h"
#include "arcane/CaseNodeNames.h"
#include "arcane/IApplication.h"
#include "arcane/IModule.h"
#include "arcane/CaseOptionBase.h"
#include "arcane/ICaseOptions.h"
#include "arcane/IService.h"
#include "arcane/IServiceInfo.h"
#include "arcane/CaseOptionError.h"
#include "arcane/CaseOptionsMulti.h"
#include "arcane/IPhysicalUnitConverter.h"
#include "arcane/IPhysicalUnitSystem.h"
#include "arcane/IStandardFunction.h"
#include "arcane/ICaseDocumentVisitor.h"
#include "arcane/MeshHandle.h"

#include <memory>
#include <vector>
#include <typeinfo>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ICaseOptionList*
createCaseOptionList(ICaseMng* m,ICaseOptions* ref_opt,XmlNode parent_element);

extern "C++" ICaseOptionList*
createCaseOptionList(ICaseOptionList* parent,ICaseOptions* ref_opt,XmlNode parent_element);

extern "C++" ICaseOptionList*
createCaseOptionList(ICaseOptionList* parent,ICaseOptions* ref_opt,XmlNode parent_element,
                     bool is_optional,bool is_multi);

extern "C++" ICaseOptionList*
createCaseOptionList(ICaseOptionsMulti* com,ICaseOptions* co,ICaseMng* m,
                     const XmlNode& element,Integer min_occurs,Integer max_occurs);

extern "C++" ICaseOptionList*
createCaseOptionList(ICaseOptionsMulti* com,ICaseOptions* co,
                     ICaseOptionList* parent,const XmlNode& element,
                     Integer min_occurs,Integer max_occurs);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void
_copyCaseOptionValue(T& out,const T& in);

template<> void _copyCaseOptionValue(String& out,const String& in) { out = in; }
template<> void _copyCaseOptionValue(bool& out,const bool& in) { out = in; }
template<> void _copyCaseOptionValue(Real& out,const Real& in) { out = in; }
template<> void _copyCaseOptionValue(Int16& out,const Int16& in) { out = in; }
template<> void _copyCaseOptionValue(Int32& out,const Int32& in) { out = in; }
template<> void _copyCaseOptionValue(Int64& out,const Int64& in) { out = in; }
template<> void _copyCaseOptionValue(Real2& out,const Real2& in) { out = in; }
template<> void _copyCaseOptionValue(Real3& out,const Real3& in) { out = in; }
template<> void _copyCaseOptionValue(Real2x2& out,const Real2x2& in) { out = in; }
template<> void _copyCaseOptionValue(Real3x3& out,const Real3x3& in) { out = in; }

template<typename T> void
_copyCaseOptionValue(UniqueArray<T>& out,const Array<T>& in)
{
  out.copy(in);
}

template<typename T> void
_copyCaseOptionValue(UniqueArray<T>& out,const UniqueArray<T>& in)
{
  out.copy(in);
}

template<typename T> void
_copyCaseOptionValue(Array<T>& out,const Array<T>& in)
{
  out.copy(in);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionName::
CaseOptionName(const String& aname)
: m_true_name(aname)
, m_translations(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionName::
CaseOptionName(const CaseOptionName& rhs)
: m_true_name(rhs.m_true_name)
, m_translations(0)
{
  if (rhs.m_translations)
    m_translations = new StringDictionary(*rhs.m_translations);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionName::
~CaseOptionName()
{
  delete m_translations;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionName::
addAlternativeNodeName(const String& lang,const String& name)
{
  if (!m_translations)
    m_translations = new StringDictionary();
  m_translations->add(lang,name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionName::
name(const String& lang) const
{
  if (!m_translations || lang.null())
    return m_true_name;
  String s = m_translations->find(lang);
  if (s.null())
    return m_true_name;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionEnumValue::
CaseOptionEnumValue(const String& name,int value)
: CaseOptionName(name)
, m_value(value)
{
}

CaseOptionEnumValue::
CaseOptionEnumValue(const CaseOptionEnumValue& rhs)
: CaseOptionName(rhs)
, m_value(rhs.m_value)
{
}

CaseOptionEnumValue::
~CaseOptionEnumValue()
{
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
  CaseOptionsPrivate(ICaseMng* cm,const String& name)
  : m_parent(nullptr), m_case_mng(cm), m_config_list(nullptr), m_module(nullptr),
    m_service_info(0), m_name(name), m_true_name(name), m_is_multi(false),
    m_is_translated_name_set(false), m_is_phase1_read(false), m_activate_function(nullptr)
  {
  }
  ~CaseOptionsPrivate()
  {
  }
 public:
  ICaseOptionList* m_parent;
  ICaseMng* m_case_mng;
  ReferenceCounter<ICaseOptionList> m_config_list;
  IModule* m_module;  //!< Module associé ou 0 s'il n'y en a pas.
  IServiceInfo* m_service_info;  //!< Service associé ou 0 s'il n'y en a pas.
  String m_name;
  String m_true_name;
  bool m_is_multi;
  bool m_is_translated_name_set;
  bool m_is_phase1_read;
  StringDictionary m_name_translations;
  ICaseFunction* m_activate_function; //!< Fonction indiquand l'état d'activation
  std::atomic<Int32> m_nb_ref = 0;
  bool m_is_case_mng_registered = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseMng* cm,const String& name)
: m_p(new CaseOptionsPrivate(cm,name))
{
  m_p->m_config_list = createCaseOptionList(cm,this,XmlNode());
  m_p->m_is_case_mng_registered = true;
  cm->registerOptions(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseOptionList* parent,const String& aname)
: m_p(new CaseOptionsPrivate(parent->caseMng(),aname))
{
  m_p->m_config_list = createCaseOptionList(parent,this,XmlNode());
  parent->addChild(this);
  m_p->m_parent = parent;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseMng* cm,const String& aname,const XmlNode& parent_elem)
: m_p(new CaseOptionsPrivate(cm,aname))
{
  m_p->m_config_list = createCaseOptionList(cm,this,parent_elem);
  m_p->m_is_case_mng_registered = true;
  cm->registerOptions(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseOptionList* parent,const String& aname,
            const XmlNode& parent_elem,bool is_optional,bool is_multi)
: m_p(new CaseOptionsPrivate(parent->caseMng(),aname))
{
  ICaseOptionList* col = createCaseOptionList(parent,this,parent_elem,is_optional,is_multi);
  m_p->m_config_list = col;
  parent->addChild(this);
  m_p->m_parent = parent;
  if (is_multi)
    _setTranslatedName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseMng* cm,const String& aname,ICaseOptionList* config_list)
: m_p(new CaseOptionsPrivate(cm,aname))
{
  m_p->m_config_list = config_list;
  m_p->m_is_case_mng_registered = true;
  cm->registerOptions(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptions::
CaseOptions(ICaseOptionList* parent,const String& aname,
            ICaseOptionList* config_list)
: m_p(new CaseOptionsPrivate(parent->caseMng(),aname))
{
  m_p->m_config_list = config_list;
  parent->addChild(this);
  m_p->m_parent = parent;
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

void CaseOptions::
addReference()
{
  ++m_p->m_nb_ref;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
removeReference()
{
  // Décrémente et retourne la valeur d'avant.
  // Si elle vaut 1, cela signifie qu'on n'a plus de références
  // sur l'objet et qu'il faut le détruire.
  Int32 v = std::atomic_fetch_add(&m_p->m_nb_ref,-1);
  if (v==1)
    delete this;
}

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
addAlternativeNodeName(const String& lang,const String& name)
{
  // On ne doit plus modifier les traductions une fois que le nom traduit
  // a été positionné. Cela peut se produire avec les services si ces
  // derniers ont une traduction dans leur axl. Dans ce cas, cette
  // dernière surcharge celle de l'option parente ce qui peut rendre
  // les noms incohérents.
  if (m_p->m_is_translated_name_set)
    return;
  m_p->m_name_translations.add(lang,name);
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
  return m_p->m_case_mng->subDomain()->defaultMeshHandle();
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

void CaseOptions::
_setTranslatedName()
{
  String lang = caseDocument()->language();
  if (m_p->m_is_translated_name_set)
    traceMng()->info() << "WARNING: translated name already set for " << m_p->m_name; 
  if (lang.null())
    m_p->m_name = m_p->m_true_name;
  else{
    String tr = m_p->m_name_translations.find(lang);
    if (!tr.null()){
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
  if (!lang.null()){
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
  bool is_phase1 = read_phase==eCaseOptionReadPhase::Phase1;
  if (is_phase1 && m_p->m_is_phase1_read)
    return;

  if (is_phase1 && !m_p->m_is_translated_name_set)
    _setTranslatedName();

  m_p->m_config_list->readChildren(is_phase1);

  if (is_phase1){
    ICaseDocument* doc = caseDocument();
    // Lit la fonction d'activation (si elle est présente)
    XmlNode velem = m_p->m_config_list->rootElement();
    CaseNodeNames* cnn = doc->caseNodeNames();
    String func_activation_name = velem.attrValue(cnn->function_activation_ref);
    if (!func_activation_name.null()){
      ICaseFunction* func = caseMng()->findFunction(func_activation_name);
      if (!func){
        CaseOptionError::addError(doc,A_FUNCINFO,velem.xpathFullName(),
                                  String::format("No function with the name '{0}' exists",
                                                 func_activation_name));
      }
      else if (func->paramType()!=ICaseFunction::ParamReal){
        CaseOptionError::addError(doc,A_FUNCINFO,velem.xpathFullName(),
                                  String::format("The function '{0}' requires a parameter of type 'time'",
                                                 func_activation_name));
      }
      else if (func->valueType()!=ICaseFunction::ValueBool){
        CaseOptionError::addError(doc,A_FUNCINFO,velem.xpathFullName(),
                                  String::format("The function '{0}' requires a parameter of type 'bool'",
                                                 func_activation_name));
      }
      else {
        m_p->m_activate_function = func;
        tm->info() << "Use the function '" << func->name() << "' to activate the option "
                    << velem.xpathFullName();
      }
    }
    // Vérifie que l'élément 'function' n'est pas présent
    {
      String func_name = velem.attrValue(cnn->function_ref);
      if (!func_name.null())
        CaseOptionError::addError(doc,A_FUNCINFO,velem.xpathFullName(),
                                  String::format("Attribute <{0}> invalid.",
                                                 cnn->function_ref));
    }
    m_p->m_is_phase1_read = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Ajoute à \a nlist les éléments non reconnus.
 */ 
void CaseOptions::
addInvalidChildren(XmlNodeList& nlist)
{
  m_p->m_config_list->addInvalidChildren(nlist);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptions::
printChildren(const String& lang,int indent)
{
  m_p->m_config_list->printChildren(lang,indent);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionsMulti::
CaseOptionsMulti(ICaseMng* cm,const String& aname,const XmlNode& parent_element,
                 Integer min_occurs,Integer max_occurs)
: CaseOptions(cm,aname,
              createCaseOptionList(this,this,cm,parent_element,
                                   min_occurs,max_occurs))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionsMulti::
CaseOptionsMulti(ICaseOptionList* parent,const String& aname,
                 const XmlNode& parent_element,
                 Integer min_occurs,Integer max_occurs)
: CaseOptions(parent,aname,
              createCaseOptionList(this,this,parent,
                                   parent_element,min_occurs,max_occurs))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionSimple::
CaseOptionSimple(const CaseOptionBuildInfo& cob)
: CaseOptionBase(cob)
, m_function(0)
, m_standard_function(0)
, m_unit_converter(0)
, m_changed_since_last_iteration(false)
, m_is_optional(cob.isOptional())
, m_has_valid_value(true)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionSimple::
CaseOptionSimple(const CaseOptionBuildInfo& cob,const String& physical_unit)
: CaseOptionBase(cob)
, m_function(nullptr)
, m_standard_function(nullptr)
, m_unit_converter(nullptr)
, m_changed_since_last_iteration(false)
, m_is_optional(cob.isOptional())
, m_has_valid_value(true)
, m_default_physical_unit(physical_unit)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionSimple::
~CaseOptionSimple()
{
  delete m_unit_converter;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionSimple::
_search(bool is_phase1)
{
  if (!is_phase1)
    return;

  //ITraceMng* msg = caseMng()->traceMng();
  const String& velem_name = name();
  XmlNodeList velems = rootElement().children(velem_name);
  XmlNode velem;
  Integer nb_elem = velems.size();
  ICaseDocument* doc = caseDocument();
  if (nb_elem>=1){
    velem = velems[0];
    if (nb_elem>=2){
      CaseOptionError::addWarning(doc,A_FUNCINFO,velem.xpathFullName(),
                                  String::format("Only one token of the element is allowed (nb_occur={0})",
                                                 nb_elem));
    }
  }

  m_element = velem;
  m_function = 0;

  _searchFunction(velem);

  String physical_unit = m_element.attrValue("unit");
  if (!physical_unit.null()){
    _setPhysicalUnit(physical_unit);
    if (_allowPhysicalUnit()){
      //TODO: VERIFIER QU'IL Y A UNE DEFAULT_PHYSICAL_UNIT.
      m_unit_converter = subDomain()->physicalUnitSystem()->createConverter(physical_unit,defaultPhysicalUnit());
    }
    else
      CaseOptionError::addError(doc,A_FUNCINFO,velem.xpathFullName(),
                                String::format("Usage of a physic unit ('{0}') is not allowed for this kind of option",
                                               physical_unit));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionSimple::
_setPhysicalUnit(const String& value)
{
  m_physical_unit = value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionSimple::
physicalUnit() const
{
  return m_physical_unit;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionSimple::
_searchFunction(XmlNode& velem)
{
  if (velem.null())
    return;

  // Recherche une éventuelle fonction associée
  String fname = caseDocument()->caseNodeNames()->function_ref;
  String func_name = velem.attrValue(fname);
  if (func_name.null())
    return;

  ICaseFunction* func = caseMng()->findFunction(func_name);
  ITraceMng* msg = caseMng()->traceMng();
  if (!func){
    msg->pfatal() << "In element <" << velem.name()
                  << ">: no function named <" << func_name << ">";
  }

  // Recherche s'il s'agit d'une fonction standard
  IStandardFunction* sf = dynamic_cast<IStandardFunction*>(func);
  if (sf){
    msg->info() << "Use standard function: " << func_name;
    m_standard_function = sf;
  }
  else{
    msg->info() << "Use function: " << func_name;
    m_function = func;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionSimple::
_setChangedSinceLastIteration(bool has_changed)
{
  m_changed_since_last_iteration = has_changed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseOptionSimple::
hasChangedSinceLastIteration() const
{
  return m_changed_since_last_iteration;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionSimple::
xpathFullName() const
{
  if (!m_element.null())
    return m_element.xpathFullName();
  String fn = rootElement().xpathFullName() + "/" + name();
  return fn;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionSimple::
defaultPhysicalUnit() const
{
  return m_default_physical_unit;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionSimple::
visit(ICaseDocumentVisitor* visitor) const
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Prise en compte du système d'unité mais uniquement pour
 * les options de type 'Real' ou 'RealArray'.
 * TODO: voir si c'est intéressant pour les autres types
 * comme Real2, Real3.
 * Pour les types intégraux comme 'Integer', il ne vaut mieux
 * pas supporter les conversions car cela risque de faire des valeurs
 * trop grandes ou nulle si la conversion donne un nombre inférieur à 1
 * (par exemple, 5 centimètres convertie en mètre donne 0).
 */
template<typename DataType> static void
_checkPhysicalConvert(IPhysicalUnitConverter* converter,DataType& value)
{
  ARCANE_UNUSED(converter);
  ARCANE_UNUSED(value);
}

static void
_checkPhysicalConvert(IPhysicalUnitConverter* converter,Real& value)
{
  if (converter){
    Real new_value = converter->convert(value);
    value = new_value;
  }
}

static void
_checkPhysicalConvert(IPhysicalUnitConverter* converter,RealUniqueArray& values)
{
  if (converter){
    //TODO utiliser tableau local pour eviter allocation
    RealUniqueArray input_values(values);
    converter->convert(input_values,values);
  }
}

template<typename DataType> static bool
_allowConvert(const DataType& value)
{
  ARCANE_UNUSED(value);
  return false;
}

static bool
_allowConvert(const Real& value)
{
  ARCANE_UNUSED(value);
  return true;
}

static bool
_allowConvert(const RealUniqueArray& value)
{
  ARCANE_UNUSED(value);
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> CaseOptionSimpleT<T>::
CaseOptionSimpleT(const CaseOptionBuildInfo& cob)
: CaseOptionSimple(cob)
{
  _copyCaseOptionValue(m_value,Type());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> CaseOptionSimpleT<T>::
CaseOptionSimpleT(const CaseOptionBuildInfo& cob,const String& physical_unit)
: CaseOptionSimple(cob,physical_unit)
{
  _copyCaseOptionValue(m_value,Type());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> bool CaseOptionSimpleT<T>::
_allowPhysicalUnit()
{
  using Type = typename CaseOptionTraitsT<T>::ContainerType;
  return _allowConvert(Type());
}

namespace
{
// Cette classe a pour but de supprimer les blancs en début et fin de
// chaîne sauf si \a Type est une 'String'.
// Si le type attendu n'est pas une 'String', on considère que les blancs
// en début et fin ne sont pas significatifs.
template<typename Type>
class StringCollapser
{
 public:
  static String collapse(const String& str)
  {
    return String::collapseWhiteSpace(str);
  }
};
template<>
class StringCollapser<String>
{
 public:
  static String collapse(const String& str)
  {
    return str;
  }
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Cherche la valeur de l'option dans le jeu de données.
 *
 * La valeur trouvée est stockée dans \a m_value.
 *
 * Si la valeur n'est pas présente dans le jeu de données, regarde s'il
 * existe une valeur par défaut et utilise cette dernière.
 */
template<typename T> void CaseOptionSimpleT<T>::
_search(bool is_phase1)
{
  CaseOptionSimple::_search(is_phase1);
  if (!is_phase1)
    return;
  ICaseDocument* doc = caseDocument();

  // Si l'option n'est pas présente dans le jeu de données, on prend
  // l'option par défaut, sauf si l'option est facultative
  String str_val = (element().null()) ? _defaultValue() : element().value();
  bool has_valid_value = true;
  if (str_val.null()){
    if (!isOptional()){
      CaseOptionError::addOptionNotFoundError(doc,A_FUNCINFO,
                                              name(),rootElement());
      return;
    }
    else
      has_valid_value = false;
  }
  _setHasValidValue(has_valid_value);
  if (has_valid_value){
    Type val = Type();
    str_val = StringCollapser<Type>::collapse(str_val);
    bool is_bad = builtInGetValue(val,str_val);
    //cerr << "** TRY CONVERT " << str_val << ' ' << val << ' ' << is_bad << endl;
    if (is_bad){
      CaseOptionError::addInvalidTypeError(doc,A_FUNCINFO,
                                           name(),rootElement(),str_val,typeToName(val));
      return;
    }
    _checkPhysicalConvert(physicalUnitConverter(),val);
    m_value = val;
  }
  _setIsInitialized();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void CaseOptionSimpleT<T>::
setDefaultValue(const Type& def_value)
{
  // Si on a une valeur donnée par l'utilisateur, on ne fait rien.
  if (isPresent())
    return;

  // Valeur déjà initialisée. Dans ce cas on remplace aussi la valeur actuelle.
  if (_isInitialized())
    m_value = def_value;

  String s;
  bool is_bad = builtInPutValue(def_value,s);
  if (is_bad)
    ARCANE_FATAL("Can not set default value");
  _setDefaultValue(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
template<typename T>
class FunctionConverterT
{
 public:
  void convert(ICaseFunction& tbl,Real t,T& value)
  {
    ARCANE_UNUSED(tbl);
    ARCANE_UNUSED(t);
    ARCANE_UNUSED(value);
    throw CaseOptionException("FunctionConverter","Invalid type");
  }
};

template<>
class FunctionConverterT<Real>
{
 public:
  void convert(ICaseFunction& tbl,Real t,Real& value)
  { tbl.value(t,value); }
};

template<>
class FunctionConverterT<Real3>
{
 public:
  void convert(ICaseFunction& tbl,Real t,Real3& value)
  { tbl.value(t,value); }
};

template<>
class FunctionConverterT<bool>
{
 public:
  void convert(ICaseFunction& tbl,Real t,bool& value)
  { tbl.value(t,value); }
};

template<>
class FunctionConverterT<Integer>
{
 public:
  void convert(ICaseFunction& tbl,Real t,Integer& value)
  { tbl.value(t,value); }
};

template<>
class FunctionConverterT<String>
{
 public:
  void convert(ICaseFunction& tbl,Real t,String& value)
  { tbl.value(t,value); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ParamType,typename ValueType>
class ComputeFunctionValue
{
 public:
  static void convert(ICaseFunction* func,ParamType t,ValueType& new_value)
  {
    FunctionConverterT<ValueType>().convert(*func,t,new_value);
  }
};

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> typename CaseOptionSimpleT<T>::Type CaseOptionSimpleT<T>::
valueAtParameter(Real t) const
{
  ICaseFunction* func = function();
  Type new_value(m_value);
  if (func){
    ComputeFunctionValue<Real,T>::convert(func,t,new_value);
    _checkPhysicalConvert(physicalUnitConverter(),new_value);
  }
  return new_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> typename CaseOptionSimpleT<T>::Type CaseOptionSimpleT<T>::
valueAtParameter(Integer t) const
{
  ICaseFunction* func = function();
  Type new_value(m_value);
  if (func){
    ComputeFunctionValue<Integer,T>::convert(func,t,new_value);
    _checkPhysicalConvert(physicalUnitConverter(),new_value);
  }
  return new_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void CaseOptionSimpleT<T>::
updateFromFunction(Real current_time,Integer current_iteration)
{
  _checkIsInitialized();
  ICaseFunction* func = function();
  if (!func)
    return;
  Type new_value(m_value);
  switch(func->paramType()){
  case ICaseFunction::ParamReal:
    ComputeFunctionValue<Real,T>::convert(func,current_time,new_value);
    break;
  case ICaseFunction::ParamInteger:
    ComputeFunctionValue<Integer,T>::convert(func,current_iteration,new_value);
    break;
  case ICaseFunction::ParamUnknown:
    break;
  }
  _checkPhysicalConvert(physicalUnitConverter(),new_value);
  this->_setChangedSinceLastIteration(m_value!=new_value);
  ITraceMng* msg = caseMng()->traceMng();
  msg->debug() << "New value for option <" << name() << "> " << new_value;
  _copyCaseOptionValue(m_value,new_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void CaseOptionSimpleT<T>::
print(const String& lang,std::ostream& o) const
{
  ARCANE_UNUSED(lang);
  _checkIsInitialized();
  if (hasValidValue())
    o << m_value;
  else
    o << "undefined";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> CaseOptionMultiSimpleT<T>::
CaseOptionMultiSimpleT(const CaseOptionBuildInfo& cob)
: CaseOptionMultiSimple(cob)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> CaseOptionMultiSimpleT<T>::
CaseOptionMultiSimpleT(const CaseOptionBuildInfo& cob,
                       const String& /*physical_unit*/)
: CaseOptionMultiSimple(cob)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> CaseOptionMultiSimpleT<T>::
~CaseOptionMultiSimpleT()
{
  const T* avalue = m_view.data();
  delete[] avalue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> bool CaseOptionMultiSimpleT<T>::
_allowPhysicalUnit()
{
  using Type = typename CaseOptionTraitsT<T>::ContainerType;
  return _allowConvert(Type());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Cherche la valeur de l'option dans le jeu de donnée.
 *
 * La valeur trouvée est stockée dans \a m_value.
 *
 * Si la valeur n'est pas présente dans le jeu de donnée, regarde s'il
 * existe une valeur par défaut et utilise cette dernière.
 */
template<typename T> void CaseOptionMultiSimpleT<T>::
_search(bool is_phase1)
{
  if (!is_phase1)
    return;
  XmlNodeList elem_list = rootElement().children(name());

  Integer asize = elem_list.size();
  _checkMinMaxOccurs(asize);
  if (asize==0)
    return;

  const Type* old_value = m_view.data();
  delete[] old_value;
  using Type = typename CaseOptionTraitsT<T>::ContainerType;
  Type* ptr_value = new Type[asize];
  m_view = ArrayViewType(asize,ptr_value);
  this->_setArray(ptr_value,asize);

  //cerr << "** MULTI SEARCH " << size << endl;
  for( Integer i=0; i<asize; ++i ){
    XmlNode velem = elem_list[i];
    // Si l'option n'est pas présente dans le jeu de donnée, on prend
    // l'option par défaut.
    String str_val = (velem.null()) ? _defaultValue() : velem.value();
    if (str_val.null())
      CaseOptionError::addOptionNotFoundError(caseDocument(),A_FUNCINFO,
                                              name(),rootElement());
    Type val    = Type();
    str_val = StringCollapser<Type>::collapse(str_val);
    bool is_bad = builtInGetValue(val,str_val);
    if (is_bad)
      CaseOptionError::addInvalidTypeError(caseDocument(),A_FUNCINFO,
                                           name(),rootElement(),str_val,typeToName(val));
    //throw CaseOptionException("get_value",name(),rootElement(),str_val,typeToName(val));
    //ptr_value[i] = val;
    _copyCaseOptionValue(ptr_value[i],val);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void CaseOptionMultiSimpleT<T>::
print(const String& lang,std::ostream& o) const
{
  ARCANE_UNUSED(lang);
  for( Integer i=0; i<this->size(); ++i )
    o << this->_ptr()[i] << " ";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void CaseOptionMultiSimpleT<T>::
visit(ICaseDocumentVisitor* visitor) const
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Cherche la valeur de l'option dans le jeu de données.
 *
 * La valeur trouvée est stockée dans \a m_value.
 *
 * Si la valeur n'est pas présente dans le jeu de données, regarde s'il
 * existe une valeur par défaut et utilise cette dernière.
 */
void CaseOptionMultiExtended::
_search(bool is_phase1)
{
  XmlNodeList elem_list = rootElement().children(name());

  Integer size = elem_list.size();
  _checkMinMaxOccurs(size);
  if (size==0)
    return;

  if (is_phase1){
    _allocate(size);
    m_values.resize(size);
  }
  else{
    //cerr << "** MULTI SEARCH " << size << endl;
    for( Integer i=0; i<size; ++i ){
      XmlNode velem = elem_list[i];
      // Si l'option n'est pas présente dans le jeu de donnée, on prend
      // l'option par défaut.
      String str_val = (velem.null()) ? _defaultValue() : velem.value();
      if (str_val.null()) {
        CaseOptionError::addOptionNotFoundError(caseDocument(),A_FUNCINFO,
                                                name(),rootElement());
        continue;
      }
      bool is_bad = _tryToConvert(str_val,i);
      if (is_bad){
        m_values[i] = String();
        CaseOptionError::addInvalidTypeError(caseDocument(),A_FUNCINFO,
                                             name(),rootElement(),str_val,_typeName());
	continue;
      }
      m_values[i] = str_val;
      //ptr_value[i] = val;
      //cerr << "** FOUND " << val << endl;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiExtended::
print(const String& lang,std::ostream& o) const
{
  ARCANE_UNUSED(lang);
  for( Integer i=0, s=_nbElem(); i<s; ++i )
    o << m_values[i] << " ";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiExtended::
visit(ICaseDocumentVisitor* visitor) const
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionExtended::
setDefaultValue(const String& def_value)
{
  // Si on a une valeur donnée par l'utilisateur, on ne fait rien.
  if (isPresent())
    return;

  // Valeur déjà initialisée. Dans ce cas on remplace aussi la valeur
  // actuelle.
  if (_isInitialized()){
    bool is_bad = _tryToConvert(def_value);
    if (is_bad){
      m_value = String();
      ARCANE_FATAL("Can not convert '{0}' to type '{1}' (option='{2}')",
                   def_value,_typeName(),xpathFullName());
    }
    m_value = def_value;
  }

  // La valeur par défaut n'a pas de langue associée.
  _setDefaultValue(def_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Cherche la valeur de l'option dans le jeu de donnée.
 *
 * La valeur trouvée est stockée dans \a m_value.
 *
 * Si la valeur n'est pas présente dans le jeu de donnée, regarde s'il
 * existe une valeur par défaut et utilise cette dernière.
 */
void CaseOptionExtended::
_search(bool is_phase1)
{
  CaseOptionSimple::_search(is_phase1);
  if (is_phase1)
    return;
  // Si l'option n'est pas présente dans le jeu de donnée, on prend
  // l'option par défaut.
  String str_val = (element().null()) ? _defaultValue() : element().value();
  bool has_valid_value = true;
  if (str_val.null()) {
    m_value = String();
    if (!isOptional()){
      CaseOptionError::addOptionNotFoundError(caseDocument(),A_FUNCINFO,
                                              name(),rootElement());
      return;
    }
    else
      has_valid_value = false;
  }
  _setHasValidValue(has_valid_value);
  if (has_valid_value){
    bool is_bad = _tryToConvert(str_val);
    if (is_bad){
      m_value = String();
      CaseOptionError::addInvalidTypeError(caseDocument(),A_FUNCINFO,
                                           name(),rootElement(),str_val,_typeName());
      return;
    }
    m_value = str_val;
  }
  _setIsInitialized();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionExtended::
print(const String& lang,std::ostream& o) const
{
  ARCANE_UNUSED(lang);
  _checkIsInitialized();
  if (hasValidValue())
    o << m_value;
  else
    o << "undefined";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionExtended::
visit(ICaseDocumentVisitor* visitor) const
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionEnumValues::
CaseOptionEnumValues()
: m_enum_values(new EnumValueList())
{
}

CaseOptionEnumValues::
~CaseOptionEnumValues()
{
  for( auto i : m_enum_values->range() ){
    delete i;
  }
  delete m_enum_values;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionEnumValue* CaseOptionEnumValues::
enumValue(Integer index) const
{
  return (*m_enum_values)[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionEnumValues::
addEnumValue(CaseOptionEnumValue* value,bool do_clone)
{
  CaseOptionEnumValue* svalue = value;
  if (do_clone)
    svalue = new CaseOptionEnumValue(*value);
  m_enum_values->add(svalue);
}

Integer CaseOptionEnumValues::
nbEnumValue() const
{
  return m_enum_values->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseOptionEnumValues::
valueOfName(const String& name,const String& lang,int& value) const
{
  for( auto ev : m_enum_values->range() ){
    const String& n = ev->name(lang);
    if (n==name){
      value = ev->value();
      return false;
    }
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionEnumValues::
nameOfValue(int value,const String& lang) const
{
  String s;
  for( auto ev : m_enum_values->range() ){
    if (ev->value()==value){
      s = ev->name(lang);
      break;
    }
  }
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionEnumValues::
getValidNames(const String& lang,StringArray& names) const
{
  for( auto ev : m_enum_values->range() ){
    names.add(ev->name(lang));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionEnum::
CaseOptionEnum(const CaseOptionBuildInfo& cob,const String& type_name)
: CaseOptionSimple(cob)
, m_type_name(type_name)
, m_enum_values(new CaseOptionEnumValues())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionEnum::
~CaseOptionEnum()
{
  delete m_enum_values;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionEnum::
_search(bool is_phase1)
{
  CaseOptionSimple::_search(is_phase1);
  if (!is_phase1)
    return;
  bool is_default = element().null();
  String str_val = (is_default) ? _defaultValue() : element().value();
  bool has_valid_value = true;
  if (str_val.null()) {
    if (!isOptional()){
      CaseOptionError::addOptionNotFoundError(caseDocument(),A_FUNCINFO,
                                              name(),rootElement());

      return;
    }
    else
      has_valid_value = false;

  }
  _setHasValidValue(has_valid_value);

  if (has_valid_value){
    String lang;
    // La valeur par défaut n'a pas de langage associé. Il ne faut donc
    // pas essayer de la convertir.
    if (!is_default)
      lang = caseDocument()->language();
    int value = 0;
    bool is_bad = m_enum_values->valueOfName(str_val,lang,value);

    if (is_bad) {
      StringUniqueArray valid_values;
      m_enum_values->getValidNames(lang,valid_values);
      CaseOptionError::addInvalidTypeError(caseDocument(),A_FUNCINFO,
                                           name(),rootElement(),str_val,m_type_name,valid_values);
      return;
    }
    _setOptionValue(value);
  }

  _setIsInitialized();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionEnum::
_setEnumDefaultValue(int def_value)
{
  // Si on a une valeur donnée par l'utilisateur, on ne fait rien.
  if (isPresent())
    return;

  // Valeur déjà initialisée. Dans ce cas on remplace aussi la valeur actuelle
  if (_isInitialized())
    _setOptionValue(def_value);

  // La valeur par défaut n'a pas de langue associée.
  _setDefaultValue(m_enum_values->nameOfValue(def_value,String()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionEnum::
print(const String& lang,std::ostream& o) const
{
  _checkIsInitialized();
  int v = _optionValue();
  o << "'" << m_enum_values->nameOfValue(v,lang) << "' (" << v << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionEnum::
visit(ICaseDocumentVisitor* visitor) const
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionEnum::
_updateFromFunction(Real current_time,Integer current_iteration)
{
  _checkIsInitialized();
  ICaseFunction* func = function();
  ITraceMng* msg = caseMng()->traceMng();
  String lang = caseDocument()->language();
  int current_value = _optionValue();
  String new_str = m_enum_values->nameOfValue(current_value,lang);
  switch(func->paramType()){
  case ICaseFunction::ParamReal:
    ComputeFunctionValue<Real,String>::convert(func,current_time,new_str);
    break;
  case ICaseFunction::ParamInteger:
    ComputeFunctionValue<Integer,String>::convert(func,current_iteration,new_str);
    break;
  case ICaseFunction::ParamUnknown:
    break;
  }
  int new_value = 0;
  bool is_bad = m_enum_values->valueOfName(new_str,lang,new_value);
  if (is_bad) {
    StringUniqueArray valid_values;
    m_enum_values->getValidNames(lang,valid_values);
    CaseOptionError::addInvalidTypeError(caseDocument(),A_FUNCINFO,
                                         name(),rootElement(),new_str,m_type_name,valid_values);
    return;
    //throw CaseOptionException("get_value",name(),rootElement(),new_str,m_type_name);
  }
  msg->debug() << "New value for enum option <" << name() << "> " << new_value;
  bool has_changed = new_value!=current_value;
  _setChangedSinceLastIteration(has_changed);
  if (has_changed)
    _setOptionValue(new_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionMultiEnum::
CaseOptionMultiEnum(const CaseOptionBuildInfo& cob,const String& type_name)
: CaseOptionBase(cob)
, m_type_name(type_name)
, m_enum_values(new CaseOptionEnumValues())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionMultiEnum::
~CaseOptionMultiEnum()
{
  delete m_enum_values;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiEnum::
_search(bool is_phase1)
{
  XmlNodeList elem_list = rootElement().children(name());

  Integer size = elem_list.size();
  _checkMinMaxOccurs(size);
  if (size==0)
    return;

  if (is_phase1){
    _allocate(size);

    const String& lang = caseDocument()->language();

    for( Integer index=0; index<size; ++index ){
      XmlNode velem = elem_list[index];
      // Si l'option n'est pas présente dans le jeu de donnée, on prend
      // l'option par défaut.
      String str_val = (velem.null()) ? _defaultValue() : velem.value();
      if (str_val.null()) {
        CaseOptionError::addOptionNotFoundError(caseDocument(),A_FUNCINFO,
                                                name(),rootElement());
        continue;
      //throw CaseOptionException("get_value",name(),rootElement());
      }

      int value = 0;
      bool is_bad = m_enum_values->valueOfName(str_val,lang,value);
      
      if (is_bad) {
        StringUniqueArray valid_values;
        m_enum_values->getValidNames(lang,valid_values);
        CaseOptionError::addInvalidTypeError(caseDocument(),A_FUNCINFO,
                                             name(),rootElement(),str_val,m_type_name,valid_values);
        continue;
      //throw CaseOptionException("get_value",name(),rootElement(),str_val,m_type_name);
      }
      else
        _setOptionValue(index,value);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiEnum::
print(const String& lang,std::ostream& o) const
{
  for( Integer i=0, s=_nbElem(); i<s; ++i ){
    int v = _optionValue(i);
    o << "'" << m_enum_values->nameOfValue(v,lang) << "' (" << v << ")";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiEnum::
visit(ICaseDocumentVisitor* visitor) const
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionComplexValue::
CaseOptionComplexValue(ICaseOptionsMulti* opt,ICaseOptionList* clist,const XmlNode& parent_elem)
: m_case_option_multi(opt)
, m_config_list(createCaseOptionList(clist,opt->toCaseOptions(),parent_elem,clist->isOptional(),true))
, m_element(parent_elem)
{
  opt->addChild(configList());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionComplexValue::
~CaseOptionComplexValue()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT ISubDomain*
_arcaneDeprecatedGetSubDomain(ICaseOptions* opt)
{
  return opt->subDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Real>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Real2>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Real3>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Real2x2>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Real3x3>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<bool>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Int16>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Int32>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Int64>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<String>;

template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<RealArray>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Real2Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Real3Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Real2x2Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Real3x3Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<BoolArray>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Int16Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Int32Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<Int64Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionSimpleT<StringArray>;

template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Real>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Real2>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Real3>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Real2x2>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Real3x3>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<bool>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Int16>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Int32>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Int64>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<String>;

template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<RealArray>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Real2Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Real3Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Real2x2Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Real3x3Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<BoolArray>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Int16Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Int32Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<Int64Array>;
template class ARCANE_TEMPLATE_EXPORT CaseOptionMultiSimpleT<StringArray>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
