// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypes.cc                                                (C) 2000-2023 */
/*                                                                           */
/* Définition des types liés aux données.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/BFloat16.h"
#include "arcane/utils/Float16.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"

#include "arcane/core/datatype/DataTypeTraits.h"
#include "arcane/core/datatype/DataTypes.h"

#include <limits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file DataTypes.h
 *
 * \brief Fichier contenant les définitions des types de données gérés par %Arcane.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" const char*
dataTypeName(eDataType type)
{
  switch(type){
  case DT_Real: return "Real";
  case DT_Int16: return "Int16";
  case DT_Int32: return "Int32";
  case DT_Int64: return "Int64";
  case DT_String: return "String";
  case DT_Real2: return "Real2";
  case DT_Real3: return "Real3";
  case DT_Real2x2: return "Real2x2";
  case DT_Real3x3: return "Real3x3";
  case DT_Byte: return "Byte";
  case DT_BFloat16: return "BFloat16";
  case DT_Float16: return "Float16";
  case DT_Float32: return "Float32";
  case DT_Unknown: return "Unknown";
  }
  return "(Invalid)";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" std::ostream&
operator<< (std::ostream& ostr,eDataType data_type)
{
  ostr << dataTypeName(data_type);
  return ostr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_DATATYPE_EXPORT eDataType
dataTypeFromName(const char* name,bool& has_error)
{
  has_error = false;
  String buf(name);
  if (buf=="Real"){
    return DT_Real;
  }
  else if (buf=="Int16"){
    return DT_Int32;
  }
  else if (buf=="Int32"){
    return DT_Int32;
  }
  else if (buf=="Int64"){
    return DT_Int64;
  }
  else if (buf=="String"){
    return DT_String;
  }
  else if (buf=="Real2"){
    return DT_Real2;
  }
  else if (buf=="Real3"){
    return DT_Real3;
  }
  else if (buf=="Real2x2"){
    return DT_Real2x2;
  }
  else if (buf=="Real3x3"){
    return DT_Real3x3;
  }
  else if (buf=="Byte"){
    return DT_Byte;
  }
  else if (buf=="BFloat16"){
    return DT_BFloat16;
  }
  else if (buf=="Float16"){
    return DT_Float16;
  }
  else if (buf=="Float32"){
    return DT_Float32;
  }
  else if (buf=="Unknown"){
    return DT_Unknown;
  }
  else
    has_error = true;
  return DT_Unknown;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_DATATYPE_EXPORT Integer
dataTypeSize(eDataType type)
{
  switch(type){
  case DT_Byte: return sizeof(Byte);
  case DT_Int16: return sizeof(Int16);
  case DT_Int32: return sizeof(Int32);
  case DT_Int64: return sizeof(Int64);
  case DT_Real: return sizeof(Real);
  case DT_Real2: return sizeof(Real2);
  case DT_Real3: return sizeof(Real3);
  case DT_Real2x2: return sizeof(Real2x2);
  case DT_Real3x3: return sizeof(Real3x3);
  case DT_BFloat16: return sizeof(BFloat16);
  case DT_Float16: return sizeof(Float16);
  case DT_Float32: return sizeof(Float32);
  case DT_String:
    throw ArgumentException("dataTypeSize()","datatype 'DT_String' has no size");
  case DT_Unknown:
    return 0;
  }
  throw ArgumentException("dataTypeSize()","Unknown datatype");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Trouve le type associé à \a name. Envoie une exception en cas d'erreur
extern "C++" ARCANE_DATATYPE_EXPORT eDataType
dataTypeFromName(const char* name)
{
  bool has_error = true;
  eDataType data_type = dataTypeFromName(name,has_error);
  if (has_error)
    throw FatalErrorException(A_FUNCINFO,String::format("Bad DataType '{0}'",name));
  return data_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" std::istream&
operator>> (std::istream& istr,eDataType& data_type)
{
  std::string buf;
  istr >> buf;
  bool has_error = true;
  data_type = dataTypeFromName(buf.c_str(),has_error);
  if (has_error){
    data_type = DT_Unknown;
    istr.setstate(std::ios_base::failbit);
  }
  return istr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static eDataInitialisationPolicy global_data_initialisation_policy = DIP_Legacy;

extern "C++" ARCANE_CORE_EXPORT void 
setGlobalDataInitialisationPolicy(eDataInitialisationPolicy init_policy)
{
  global_data_initialisation_policy = init_policy;
}

extern "C++" ARCANE_CORE_EXPORT eDataInitialisationPolicy
getGlobalDataInitialisationPolicy()
{
  return global_data_initialisation_policy;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static Real _getNan()
{
  return std::numeric_limits<Real>::signaling_NaN();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Type> static void
_fillNoNan(ArrayView<Type> ptr)
{
  Type v = Type();
  Integer n = ptr.size();
  for( Integer i=0; i<n; ++i )
    ptr[i] = v;
}

void DataTypeTraitsT<Byte>::
fillNan(ArrayView<Type> ptr)
{
  _fillNoNan(ptr);
}

void DataTypeTraitsT<Int16>::
fillNan(ArrayView<Type> ptr)
{
  _fillNoNan(ptr);
}

void DataTypeTraitsT<Int32>::
fillNan(ArrayView<Type> ptr)
{
  _fillNoNan(ptr);
}

void DataTypeTraitsT<Int64>::
fillNan(ArrayView<Type> ptr)
{
  _fillNoNan(ptr);
}

void DataTypeTraitsT<String>::
fillNan(ArrayView<Type> ptr)
{
  _fillNoNan(ptr);
}

static void
_fillWithNan(RealArrayView ptr)
{
  Real v = _getNan();
  Integer n = ptr.size();
  for( Integer i=0; i<n; ++i )
    ptr[i] = v;
}

void DataTypeTraitsT<Real>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(ptr);
}

void DataTypeTraitsT<Real2>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(RealArrayView(ptr.size()*2,(Real*)ptr.data()));
}

void DataTypeTraitsT<Real2x2>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(RealArrayView(ptr.size()*4,(Real*)ptr.data()));
}

void DataTypeTraitsT<Real3x3>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(RealArrayView(ptr.size()*9,(Real*)ptr.data()));
}

void DataTypeTraitsT<Real3>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(RealArrayView(ptr.size()*3,(Real*)ptr.data()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto DataTypeTraitsT<String>::
defaultValue() -> String
{
  return String();
}

auto DataTypeTraitsT<Real3>::
defaultValue() -> Real3
{
  return Real3::zero();
}

auto DataTypeTraitsT<Real3x3>::
defaultValue() -> Real3x3
{
  return Real3x3::zero();
}

auto DataTypeTraitsT<Real2>::
defaultValue() -> Real2
{
  return Real2::null();
}

auto DataTypeTraitsT<Real2x2>::
defaultValue() -> Real2x2
{
  return Real2x2::null();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

