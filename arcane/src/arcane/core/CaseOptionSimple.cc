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

#include "arcane/CaseOptionSimple.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/ParameterCaseOption.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/CaseOptionException.h"
#include "arcane/core/CaseOptionBuildInfo.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/ICaseFunction.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/CaseNodeNames.h"
#include "arcane/core/CaseOptionError.h"
#include "arcane/core/IPhysicalUnitConverter.h"
#include "arcane/core/IPhysicalUnitSystem.h"
#include "arcane/core/IStandardFunction.h"
#include "arcane/core/ICaseDocumentVisitor.h"
#include "arcane/core/internal/StringVariableReplace.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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

CaseOptionSimple::
CaseOptionSimple(const CaseOptionBuildInfo& cob)
: CaseOptionBase(cob)
, m_is_optional(cob.isOptional())
, m_has_valid_value(true)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionSimple::
CaseOptionSimple(const CaseOptionBuildInfo& cob,const String& physical_unit)
: CaseOptionBase(cob)
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
  ICaseDocumentFragment* doc = caseDocumentFragment();
  if (nb_elem>=1){
    velem = velems[0];
    if (nb_elem>=2){
      CaseOptionError::addWarning(doc,A_FUNCINFO,velem.xpathFullName(),
                                  String::format("Only one token of the element is allowed (nb_occur={0})",
                                                 nb_elem));
    }
  }

  // Liste des options de la ligne de commande.
  {
    const ParameterList& params = caseMng()->application()->applicationInfo().commandLineArguments().parameters();
    const ParameterCaseOption pco{ params.getParameterCaseOption(doc->language()) };

    String reference_input = pco.getParameterOrNull(String::format("{0}/{1}", rootElement().xpathFullName(), velem_name), 1, false);
    if (!reference_input.null()) {
      // Si l'utilisateur a spécifié une option qui n'est pas présente dans le
      // jeu de données, on doit la créer.
      if (velem.null()) {
        velem = rootElement().createElement(name());
      }
      velem.setValue(reference_input);
    }
    if (!velem.null()) {
      velem.setValue(StringVariableReplace::replaceWithCmdLineArgs(params, velem.value(), true));
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
      m_unit_converter = caseMng()->physicalUnitSystem()->createConverter(physical_unit,defaultPhysicalUnit());
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
  String fname = caseDocumentFragment()->caseNodeNames()->function_ref;
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
  ICaseDocumentFragment* doc = caseDocumentFragment();

  // Si l'option n'est pas présente dans le jeu de données, on prend
  // l'option par défaut, sauf si l'option est facultative
  String str_val = (_element().null()) ? _defaultValue() : _element().value();
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
template <typename T>
void CaseOptionMultiSimpleT<T>::
_search(bool is_phase1)
{
  if (!is_phase1)
    return;

  const ParameterList& params = caseMng()->application()->applicationInfo().commandLineArguments().parameters();
  const ParameterCaseOption pco{ params.getParameterCaseOption(caseDocumentFragment()->language()) };

  String full_xpath = String::format("{0}/{1}", rootElement().xpathFullName(), name());
  // !!! En XML, on commence par 1 et non 0.
  UniqueArray<Integer> option_in_param;

  pco.indexesInParam(full_xpath, option_in_param, false);

  XmlNodeList elem_list = rootElement().children(name());
  Integer asize = elem_list.size();

  bool is_optional = isOptional();

  if (asize == 0 && option_in_param.empty() && is_optional) {
    return;
  }

  Integer min_occurs = minOccurs();
  Integer max_occurs = maxOccurs();

  Integer max_in_param = 0;

  if (!option_in_param.empty()) {
    max_in_param = option_in_param[0];
    for (Integer index : option_in_param) {
      if (index > max_in_param)
        max_in_param = index;
    }
    if (max_occurs >= 0) {
      if (max_in_param > max_occurs) {
        ARCANE_FATAL("Max in param > max_occurs");
      }
    }
  }

  if (max_occurs >= 0) {
    if (asize > max_occurs) {
      ARCANE_FATAL("Nb in XmlNodeList > max_occurs");
    }
  }

  Integer final_size = std::max(asize, std::max(min_occurs, max_in_param));

  const Type* old_value = m_view.data();
  delete[] old_value;
  using Type = typename CaseOptionTraitsT<T>::ContainerType;
  Type* ptr_value = new Type[final_size];
  m_view = ArrayViewType(final_size, ptr_value);
  this->_setArray(ptr_value, final_size);

  //cerr << "** MULTI SEARCH " << size << endl;
  for (Integer i = 0; i < final_size; ++i) {
    String str_val;

    if (option_in_param.contains(i + 1)) {
      str_val = pco.getParameterOrNull(full_xpath, i + 1, false);
    }
    else if (i < asize) {
      XmlNode velem = elem_list[i];
      if (!velem.null()) {
        str_val = velem.value();
      }
    }

    if (str_val.null()) {
      str_val = _defaultValue();
    }
    else {
      // Dans un else : Le remplacement de symboles ne s'applique pas pour les valeurs par défault du .axl.
      str_val = StringVariableReplace::replaceWithCmdLineArgs(params, str_val, true);
    }

    if (str_val.null())
      CaseOptionError::addOptionNotFoundError(caseDocumentFragment(), A_FUNCINFO,
                                              name(), rootElement());
    Type val = Type();
    str_val = StringCollapser<Type>::collapse(str_val);
    bool is_bad = builtInGetValue(val, str_val);
    if (is_bad)
      CaseOptionError::addInvalidTypeError(caseDocumentFragment(), A_FUNCINFO,
                                           name(), rootElement(), str_val, typeToName(val));
    //throw CaseOptionException("get_value",name(),rootElement(),str_val,typeToName(val));
    //ptr_value[i] = val;
    _copyCaseOptionValue(ptr_value[i], val);
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

String CaseOptionSimple::
_convertFunctionRealToString(ICaseFunction* func,Real t)
{
  String v;
  ComputeFunctionValue<Real,String>::convert(func,t,v);
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionSimple::
_convertFunctionIntegerToString(ICaseFunction* func,Integer t)
{
  String v;
  ComputeFunctionValue<Integer,String>::convert(func,t,v);
  return v;
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
