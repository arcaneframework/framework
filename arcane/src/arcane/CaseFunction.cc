// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseFunction.cc                                             (C) 2000-2017 */
/*                                                                           */
/* Classe gérant une fonction du jeu de données.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/MathUtils.h"

#include "arcane/CaseFunction.h"
#include "arcane/ISubDomain.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! \deprecated Utiliser CaseFunctionBuildInfo(ITraceMng* tm,const String& name)
CaseFunctionBuildInfo::
CaseFunctionBuildInfo(ISubDomain* sd,const String& name)
: CaseFunctionBuildInfo(sd->traceMng(),name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseFunction::
CaseFunction(const CaseFunctionBuildInfo& info)
: m_trace(info.m_trace_mng)
, m_name(info.m_name)
, m_param_type(info.m_param_type)
, m_value_type(info.m_value_type)
, m_transform_param_func(info.m_transform_param_func)
, m_transform_value_func(info.m_transform_value_func)
, m_deltat_coef(info.m_deltat_coef)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseFunction::
~CaseFunction()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunction::
setName(const String& new_name)
{
  if (new_name.null())
    return;
  m_name = new_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunction::
setParamType(eParamType new_type)
{
  if (new_type==m_param_type)
    return;
  m_param_type = new_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunction::
setValueType(eValueType new_type)
{
  if (new_type==m_value_type)
    return;
  m_value_type = new_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunction::
setTransformValueFunction(const String& str)
{
  m_transform_value_func = str;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunction::
setTransformParamFunction(const String& str)
{
  m_transform_param_func = str;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseFunction::
checkIfValid() const
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real CaseFunction::
_applyValueComulTransform(Real v,Real comul) const
{
  return v * comul;
}
Integer CaseFunction::
_applyValueComulTransform(Integer v,Integer comul) const
{
  return v * comul;
}
Real3 CaseFunction::
_applyValueComulTransform(Real3 v,Real3 comul) const
{
  return v * comul;
}
String CaseFunction::
_applyValueComulTransform(const String& v,const String& comul) const
{
  ARCANE_UNUSED(v);
  ARCANE_UNUSED(comul);
  ARCANE_FATAL("Invalid for type 'String'");
}
bool CaseFunction::
_applyValueComulTransform(bool v,bool comul) const
{
  ARCANE_UNUSED(v);
  ARCANE_UNUSED(comul);
  ARCANE_FATAL("Invalid for type 'bool'");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ValueType> void CaseFunction::
_applyValueTransform2(ValueType& value) const
{
  // Applique la transformation...
  // Pour l'instant, uniquement un coefficient multiplicateur.
  if (m_transform_value_func.null())
    return;
  ValueType comul = ValueType();
  bool is_bad = builtInGetValue(comul,m_transform_value_func);
  if (is_bad)
    ARCANE_FATAL("Can not convert 'comul' value '{0}'",m_transform_value_func);
  value = _applyValueComulTransform(value,comul);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunction::
_applyValueTransform(Real& value) const
{
  _applyValueTransform2(value);
}
void CaseFunction::
_applyValueTransform(Real3& value) const
{
  _applyValueTransform2(value);
}
void CaseFunction::
_applyValueTransform(Integer& value) const
{
  _applyValueTransform2(value);
}
void CaseFunction::
_applyValueTransform(String& value) const
{
  _applyValueTransform2(value);
}
void CaseFunction::
_applyValueTransform(bool& value) const
{
  _applyValueTransform2(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ParamType> void CaseFunction::
_applyParamTransform2(ParamType& param) const
{
  // Applique la transformation...
  // Pour l'instant, uniquement un coefficient multiplicateur.
  if (m_transform_param_func.null())
    return;
  
  ParamType comul = ParamType();
  bool is_bad = builtInGetValue(comul,m_transform_param_func);
  if (is_bad)
    ARCANE_FATAL("Can not convert 'comul-x' value '{0}'",m_transform_param_func);
  if (math::isZero(comul))
    ARCANE_FATAL("The parameter 'comul-x' can not be zero");
  param = param / comul;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunction::
_applyParamTransform(Real& value) const
{
  _applyParamTransform2(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunction::
_applyParamTransform(Integer& value) const
{
  _applyParamTransform2(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

