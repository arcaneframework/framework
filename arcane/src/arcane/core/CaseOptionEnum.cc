// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionEnum.cc                                           (C) 2000-2023 */
/*                                                                           */
/* Option du jeu de données de type énuméré.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionEnum.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/ICaseFunction.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/StringDictionary.h"
#include "arcane/core/CaseOptionError.h"
#include "arcane/core/ICaseDocumentVisitor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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

CaseOptionEnumValues::
CaseOptionEnumValues()
: m_enum_values(new EnumValueList())
{
}

CaseOptionEnumValues::
~CaseOptionEnumValues()
{
  for( auto i : (*m_enum_values) ){
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
  for( auto ev : (*m_enum_values) ){
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
  for( auto ev : (*m_enum_values) ){
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
  for( auto ev : (*m_enum_values) ){
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
  bool is_default = _element().null();
  String str_val = (is_default) ? _defaultValue() : _element().value();
  bool has_valid_value = true;
  if (str_val.null()) {
    if (!isOptional()){
      CaseOptionError::addOptionNotFoundError(caseDocumentFragment(),A_FUNCINFO,
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
      lang = caseDocumentFragment()->language();
    int value = 0;
    bool is_bad = m_enum_values->valueOfName(str_val,lang,value);

    if (is_bad) {
      StringUniqueArray valid_values;
      m_enum_values->getValidNames(lang,valid_values);
      CaseOptionError::addInvalidTypeError(caseDocumentFragment(),A_FUNCINFO,
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
  String lang = caseDocumentFragment()->language();
  int current_value = _optionValue();
  String new_str = m_enum_values->nameOfValue(current_value,lang);
  switch(func->paramType()){
  case ICaseFunction::ParamReal:
    new_str = _convertFunctionRealToString(func,current_time);
    break;
  case ICaseFunction::ParamInteger:
    new_str = _convertFunctionIntegerToString(func,current_iteration);
    break;
  case ICaseFunction::ParamUnknown:
    break;
  }
  int new_value = 0;
  bool is_bad = m_enum_values->valueOfName(new_str,lang,new_value);
  if (is_bad) {
    StringUniqueArray valid_values;
    m_enum_values->getValidNames(lang,valid_values);
    CaseOptionError::addInvalidTypeError(caseDocumentFragment(),A_FUNCINFO,
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

    const String& lang = caseDocumentFragment()->language();

    for( Integer index=0; index<size; ++index ){
      XmlNode velem = elem_list[index];
      // Si l'option n'est pas présente dans le jeu de donnée, on prend
      // l'option par défaut.
      String str_val = (velem.null()) ? _defaultValue() : velem.value();
      if (str_val.null()) {
        CaseOptionError::addOptionNotFoundError(caseDocumentFragment(),A_FUNCINFO,
                                                name(),rootElement());
        continue;
      //throw CaseOptionException("get_value",name(),rootElement());
      }

      int value = 0;
      bool is_bad = m_enum_values->valueOfName(str_val,lang,value);
      
      if (is_bad) {
        StringUniqueArray valid_values;
        m_enum_values->getValidNames(lang,valid_values);
        CaseOptionError::addInvalidTypeError(caseDocumentFragment(),A_FUNCINFO,
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
