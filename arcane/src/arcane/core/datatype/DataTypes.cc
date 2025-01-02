// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypes.cc                                                (C) 2000-2024 */
/*                                                                           */
/* Définition des types liés aux données.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NumericTypes.h"
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

namespace
{
  const char* N_ALL_NAMES[NB_ARCANE_DATA_TYPE] = {
    DataTypeNames::N_BYTE, DataTypeNames::N_REAL,
    DataTypeNames::N_INT16, DataTypeNames::N_INT32, DataTypeNames::N_INT64,
    DataTypeNames::N_STRING,
    DataTypeNames::N_REAL2, DataTypeNames::N_REAL3, DataTypeNames::N_REAL2x2, DataTypeNames::N_REAL3x3,
    DataTypeNames::N_BFLOAT16, DataTypeNames::N_FLOAT16, DataTypeNames::N_FLOAT32,
    DataTypeNames::N_INT8, DataTypeNames::N_FLOAT128, DataTypeNames::N_INT128,
    DataTypeNames::N_UNKNOWN
  };

  //! Taille d'un élément du type
  int ALL_SIZEOF[NB_ARCANE_DATA_TYPE] = {
    sizeof(Byte), sizeof(Real),
    sizeof(Int16), sizeof(Int32), sizeof(Int64),
    -1,
    sizeof(Real2), sizeof(Real3), sizeof(Real2x2), sizeof(Real3x3),
    sizeof(BFloat16), sizeof(Float16), sizeof(Float32),
    sizeof(Int8), sizeof(Float128), sizeof(Int128),
    0
  };
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" const char*
dataTypeName(eDataType type)
{
  Int32 v = type;
  if (v >= NB_ARCANE_DATA_TYPE)
    return "(Invalid)";
  return N_ALL_NAMES[v];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" std::ostream&
operator<<(std::ostream& ostr, eDataType data_type)
{
  ostr << dataTypeName(data_type);
  return ostr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_DATATYPE_EXPORT eDataType
dataTypeFromName(const char* name, bool& has_error)
{
  has_error = false;
  std::string_view buf(name);
  for (int i = 0; i < NB_ARCANE_DATA_TYPE; ++i) {
    if (buf == std::string_view(N_ALL_NAMES[i])) {
      has_error = false;
      return static_cast<eDataType>(i);
    }
  }
  return DT_Unknown;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_DATATYPE_EXPORT Integer
dataTypeSize(eDataType type)
{
  if (type == DT_String)
    ARCANE_THROW(ArgumentException, "datatype 'DT_String' has no size");
  const Int32 v = type;
  if (v >= NB_ARCANE_DATA_TYPE)
    ARCANE_THROW(ArgumentException, "Invalid datatype value '{0}'", v);
  return ALL_SIZEOF[v];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Trouve le type associé à \a name. Envoie une exception en cas d'erreur
extern "C++" ARCANE_DATATYPE_EXPORT eDataType
dataTypeFromName(const char* name)
{
  bool has_error = true;
  eDataType data_type = dataTypeFromName(name, has_error);
  if (has_error)
    ARCANE_FATAL("Bad DataType '{0}'", name);
  return data_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" std::istream&
operator>>(std::istream& istr, eDataType& data_type)
{
  std::string buf;
  istr >> buf;
  bool has_error = true;
  data_type = dataTypeFromName(buf.c_str(), has_error);
  if (has_error) {
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

namespace
{
  Real _getNan()
  {
    return std::numeric_limits<Real>::signaling_NaN();
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Type> static void
_fillNoNan(ArrayView<Type> ptr)
{
  Type v = Type();
  Integer n = ptr.size();
  for (Integer i = 0; i < n; ++i)
    ptr[i] = v;
}

void DataTypeTraitsT<Byte>::
fillNan(ArrayView<Type> ptr)
{
  _fillNoNan(ptr);
}

void DataTypeTraitsT<Int8>::
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

template<typename Type> static void
_fillWithNan(ArrayView<Type> ptr)
{
  Real v = _getNan();
  Integer n = ptr.size();
  for (Integer i = 0; i < n; ++i)
    ptr[i] = v;
}

void DataTypeTraitsT<Real>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(ptr);
}

void DataTypeTraitsT<Float32>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(ptr);
}

void DataTypeTraitsT<BFloat16>::
fillNan(ArrayView<Type> ptr)
{
  _fillNoNan(ptr);
}

void DataTypeTraitsT<Float16>::
fillNan(ArrayView<Type> ptr)
{
  _fillNoNan(ptr);
}

void DataTypeTraitsT<Real2>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(RealArrayView(ptr.size() * 2, (Real*)ptr.data()));
}

void DataTypeTraitsT<Real2x2>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(RealArrayView(ptr.size() * 4, (Real*)ptr.data()));
}

void DataTypeTraitsT<Real3x3>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(RealArrayView(ptr.size() * 9, (Real*)ptr.data()));
}

void DataTypeTraitsT<Real3>::
fillNan(ArrayView<Type> ptr)
{
  _fillWithNan(RealArrayView(ptr.size() * 3, (Real*)ptr.data()));
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

