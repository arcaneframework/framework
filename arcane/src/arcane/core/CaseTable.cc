// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseTable.cc                                                (C) 2000-2023 */
/*                                                                           */
/* Classe gérant une table de marche.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/CaseTable.h"
#include "arcane/CaseTableParams.h"
#include "arcane/MathUtils.h"
#include "arcane/ISubDomain.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T> void
_verboseBuiltInGetValue(const CaseTable* table,Integer index,T& v,const String& s)
{
  bool is_bad = builtInGetValue(v,s);
  if (is_bad){
    ARCANE_FATAL("Table '{0}' index={1} : can not convert value '{2}' to type '{3}'",
                 table->name(),index,s,typeToName(v));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseTable::
CaseTable(const CaseFunctionBuildInfo& info, eCurveType curve_type)
: CaseFunction(info)
, m_param_list(nullptr)
, m_curve_type(curve_type)
{
  m_param_list = new CaseTableParams(info.m_param_type);
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_LINEAR_SEARCH_IN_CASE_TABLE", true))
    m_use_fast_search = v.value() == 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseTable::
~CaseTable()
{
  delete m_param_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseTable::
_isValidIndex(Integer index) const
{
  Integer n = nbElement();
  if (n==0)
    return false;
  if (index>=n)
    return false;
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseTable::
valueToString(Integer id,String& str) const
{
  if (_isValidIndex(id))
    str = m_value_list[id].asString();
  else
    str = String();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseTable::
paramToString(Integer id,String& str) const
{
  m_param_list->toString(id,str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseTable::
value(Real param,Real& v) const
{ _findValueAndApplyTransform(param,v); }

void CaseTable::
value(Real param,Integer& v) const
{ _findValueAndApplyTransform(param,v); }

void CaseTable::
value(Real param,bool& v) const
{ _findValueAndApplyTransform(param,v); }

void CaseTable::
value(Real param,String& v) const
{ _findValueAndApplyTransform(param,v); }

void CaseTable::
value(Real param,Real3& v) const
{ _findValueAndApplyTransform(param,v); }

void CaseTable::
value(Integer param,Real& v) const
{ _findValueAndApplyTransform(param,v); }

void CaseTable::
value(Integer param,Integer& v) const
{ _findValueAndApplyTransform(param,v); }

void CaseTable::
value(Integer param,bool& v) const
{ _findValueAndApplyTransform(param,v); }

void CaseTable::
value(Integer param,String& v) const
{ _findValueAndApplyTransform(param,v); }

void CaseTable::
value(Integer param,Real3& v) const
{ _findValueAndApplyTransform(param,v); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseTable::eError CaseTable::
setParam(Integer id,const String& str)
{
  return m_param_list->setValue(id,str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseTable::eError CaseTable::
setValue(Integer id,const String& str)
{
  if (_isValidIndex(id))
    return _setValue(id,str);
  return ErrNo;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseTable::eError CaseTable::
_setValue(Integer index,const String& value_str)
{
  SmallVariant variant(value_str);
  eValueType value_type(valueType());
  if (value_type==ValueReal){
    Real v = 0.; 
    if (builtInGetValue(v,value_str)){
      return ErrCanNotConvertValueToRightType;
    }
    variant.setValueAll(v);
  }
  else if (value_type==ValueInteger){
    Integer v = 0;
    if (builtInGetValue(v,value_str)){
      return ErrCanNotConvertValueToRightType;
    }
    variant.setValueAll(v);
  }
  else if (value_type==ValueBool){
    bool v = 0;
    if (builtInGetValue(v,value_str)){
      return ErrCanNotConvertValueToRightType;
    }
    variant.setValueAll(v);
  }

  if (index>=m_value_list.size()){
    m_value_list.add(variant);
  }
  else
    m_value_list[index] = variant;

  return ErrNo;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseTable::
setParamType(eParamType new_type)
{
  bool type_changed = (new_type!=paramType());
  CaseFunction::setParamType(new_type);
  if (type_changed)
    m_param_list->setType(new_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo en cas d'erreur de l'un des deux, ne pas changer la valeur de l'autre.
 */
CaseTable::eError CaseTable::
appendElement(const String& param,const String& value)
{
  eError err = m_param_list->appendValue(param);
  if (err!=ErrNo)
    return err;

  return _setValue(m_value_list.size(),value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseTable::
insertElement(Integer id)
{
  // Ajoute un élément à la fin.
  Integer n = nbElement();
  if (n==0)
    return;
  if (id>=n)
    id = n-1;

  
  String param_str;
  m_param_list->toString(n-1,param_str);
  m_param_list->appendValue(param_str);

  //SmallVariant value_str(m_value_list[n-1]);
  m_value_list.add(m_value_list[n-1]);

  for( Integer i=n; i>id; --i ){
    m_param_list->toString(i-1,param_str);
    m_param_list->setValue(i,param_str);
    m_value_list[i] = m_value_list[i-1];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseTable::
removeElement(Integer id)
{
  if (!_isValidIndex(id))
    return;
  
  m_value_list.remove(id);
  m_param_list->removeValue(id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CaseTable::
nbElement() const
{
  return m_param_list->nbElement();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseTable::
checkIfValid() const
{
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class U,class V> void CaseTable::
_findValueAndApplyTransform(U param,V& avalue) const
{
  _findValue(param,avalue);
  _applyValueTransform(avalue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename U> Real
_curveLinear(Real current_value,Real next_value,U t)
{
  return (Real)(current_value + (next_value-current_value)*t );
}
template<typename U> Real3
_curveLinear(Real3 current_value,Real3 next_value,U t)
{
  return current_value + (next_value-current_value)*t;
}
template<typename U> Integer
_curveLinear(Integer current_value,Integer next_value,U t)
{
  return Convert::toInteger( current_value + (next_value-current_value)*t );
}
template<typename U> bool
_curveLinear(bool,bool,U)
{
  ARCANE_FATAL("Invalid for 'bool' type");
}
template<typename U> String
_curveLinear(const String&,const String&,U)
{
  ARCANE_FATAL("Invalid for 'String' type");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul la valeur de la courbe pour le paramètre \a param.
 * La valeur est stockée dans \a value.
 *
 * \retval false en cas de succès
 * \retval true si erreur
 */
template<typename ParamType,typename ValueType> void CaseTable::
_findValue(ParamType param,ValueType& avalue) const
{
  _applyParamTransform(param);

  // On suppose que les éléments de la table sont rangés par paramètres
  // croissants (begin[i+1]>begin[i])
  Integer nb_elem = nbElement();

  Int32 i0 = 0;
  Int32 iend = nb_elem;
  if (m_use_fast_search)
    m_param_list->getRange(param,i0,iend);

  for( Integer i=i0; i<iend; ++i ){
    const String& current_value_str = m_value_list[i].asString();
    ParamType current_begin_value;
    m_param_list->value(i,current_begin_value);
		
    // Si dernier élément, on prend la valeur de celui-ci
    if ((i+1)==nb_elem){
      _verboseBuiltInGetValue(this,i,avalue,current_value_str);
      return;
    }
    
    // Tout d'abord, regarde si le paramètre n'est pas
    // égal ou inférieur à la borne actuelle.
    if (math::isEqual(current_begin_value,param) || param<current_begin_value ){
      _verboseBuiltInGetValue(this,i,avalue,current_value_str);
      return;
    }

    const String& next_value_str = m_value_list[i+1].asString();
    ParamType next_begin_value;
    m_param_list->value(i+1,next_begin_value);

    // Regarde si le paramètre n'est pas égal à la borne de l'élément suivant.
    if (math::isEqual(next_begin_value,param)){
      _verboseBuiltInGetValue(this,i,avalue,next_value_str);
      return;
    }

    // Regarde si le paramètre est compris entre l'élément courant et le suivant.
    if (param>current_begin_value && param<next_begin_value){
      ValueType current_value = ValueType();
      ValueType next_value = ValueType();
      _verboseBuiltInGetValue(this,i,current_value,current_value_str);
      if (m_curve_type==CurveConstant){
        avalue = current_value;
        return;
      }
      if (m_curve_type==CurveLinear){
        _verboseBuiltInGetValue(this,i,next_value,next_value_str);
        ParamType diff = next_begin_value - current_begin_value;
        if (math::isZero(diff))
          ARCANE_FATAL("Table '{0}' index={1} : DeltaX==0.0",name(),i);
        ParamType t = (param - current_begin_value) / diff;
        avalue = _curveLinear<ParamType>(current_value,next_value,t);
        return;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

