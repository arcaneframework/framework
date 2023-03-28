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

#include "arcane/core/CaseOptionsMulti.h"

#include <atomic>

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


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
class CaseOptionsPrivate
{
 public:

  CaseOptionsPrivate(ICaseMng* cm,const String& name)
  : m_case_mng(cm), m_name(name), m_true_name(name)
  , m_mesh_handle(cm->meshMng()->defaultMeshHandle())
  {
  }

  CaseOptionsPrivate(ICaseOptionList* co_list,const String& name)
  : m_case_mng(co_list->caseMng()), m_name(name), m_true_name(name),
    m_mesh_handle(co_list->meshHandle())
  {
    if (m_mesh_handle.isNull())
      m_mesh_handle = m_case_mng->meshMng()->defaultMeshHandle();
  }

 public:

  ICaseOptionList* m_parent = nullptr;
  ICaseMng* m_case_mng;
  ReferenceCounter<ICaseOptionList> m_config_list;
  IModule* m_module = nullptr;  //!< Module associé ou 0 s'il n'y en a pas.
  IServiceInfo* m_service_info = nullptr;  //!< Service associé ou 0 s'il n'y en a pas.
  String m_name;
  String m_true_name;
  bool m_is_multi = false;
  bool m_is_translated_name_set = false;
  bool m_is_phase1_read = false;
  StringDictionary m_name_translations;
  ICaseFunction* m_activate_function = nullptr; //!< Fonction indiquand l'état d'activation
  std::atomic<Int32> m_nb_ref = 0;
  bool m_is_case_mng_registered = false;
  MeshHandle m_mesh_handle;
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
: m_p(new CaseOptionsPrivate(parent,aname))
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
: m_p(new CaseOptionsPrivate(parent,aname))
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

void CaseOptions::
_setMeshHandle(const MeshHandle& handle)
{
  traceMng()->info(5) << "SetMeshHandle for " << m_p->m_name << " mesh_name=" << handle.meshName();
  m_p->m_mesh_handle = handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne le maillage associé à cette option.
 *
 * Si \a mesh_name est nul ou vide alors le maillage associé à cette
 * option est celui de l'option parente. Si l'option n'a pas de parent alors
 * c'est le maillage par défaut.
 *
 * Si \a mesh_name n'est pas nul, il y a deux possibilités:
 * - si le maillage spécifié existe alors l'option sera associée à ce maillage
 * - s'il n'existe pas, alors l'option est désactivée et les éventuelles options
 * filles ne seront pas lues. Ce dernier cas arrive par exemple si un service
 * est associé à un maillage supplémentaire mais que ce dernier est optionnel.
 * Dans ce cas l'option ne doit pas être lue.
 *
 * \retval true si l'option est désactivée suite à cet appel.
 */
bool CaseOptions::
_setMeshHandleAndCheckDisabled(const String& mesh_name)
{
  if (mesh_name.empty()){
    // Mon maillage est celui de mon parent
    if (m_p->m_parent)
      _setMeshHandle(m_p->m_parent->meshHandle());
  }
  else{
    // Un maillage différent du maillage par défaut est associé à l'option.
    // Récupère le MeshHandle associé s'il n'existe. S'il n'y en a pas on
    // désactive l'option.
    // Si aucun maillage du nom de celui qu'on cherche n'existe, n'alloue pas le service
    MeshHandle* handle = caseMng()->meshMng()->findMeshHandle(mesh_name,false);
    if (!handle){
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
  String str_val = (_element().null()) ? _defaultValue() : _element().value();
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

CaseOptionComplexValue::
CaseOptionComplexValue(ICaseOptionsMulti* opt,ICaseOptionList* clist,const XmlNode& parent_elem)
: m_config_list(createCaseOptionList(clist,opt->toCaseOptions(),parent_elem,clist->isOptional(),true))
, m_element(parent_elem)
{
  opt->addChild(_configList());
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
