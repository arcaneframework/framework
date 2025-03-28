// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicDataType.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Définition des types liés aux données.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BasicDataType.h"
#include "arccore/base/String.h"
#include "arccore/base/ArgumentException.h"
#include "arccore/base/FatalErrorException.h"

#include <limits>
#include <string>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file BasicDataType.h
 *
 * \brief Fichier contenant les définitions des types de données basiques
 * gérés par %Arccore.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace
{
const char* N_BYTE = "Byte";
const char* N_BFLOAT16 = "BFloat16";
const char* N_FLOAT16 = "Float16";
const char* N_FLOAT32 = "Float32";
const char* N_FLOAT64 = "Float64";
const char* N_FLOAT128 = "Float128";
const char* N_INT8 = "Int8";
const char* N_INT16 = "Int16";
const char* N_INT32 = "Int32";
const char* N_INT64 = "Int64";
const char* N_INT128 = "Int128";
const char* N_UNKNOWN = "Unknown";
const char* N_INVALID = "Invalid";

//! Nom des types. Doit correspondre aux valeurs de l'énumération eBasicDataType
const char* N_ALL_NAMES[NB_BASIC_DATA_TYPE] =
  {
    N_UNKNOWN, N_BYTE,
    N_FLOAT16, N_FLOAT32, N_FLOAT64, N_FLOAT128,
    N_INT16, N_INT32, N_INT64, N_INT128,
    N_BFLOAT16, N_INT8
  };

//! Taille d'un élément du type
int ALL_SIZEOF[NB_BASIC_DATA_TYPE] =
  {
    0, 1,
    2, 4, 8, 16,
    2, 4, 8, 16,
    2, 1
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" const char*
basicDataTypeName(eBasicDataType type)
{
  Byte b = (Byte)type;
  if (b>=NB_BASIC_DATA_TYPE)
    return N_INVALID;
  return N_ALL_NAMES[b];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" std::ostream&
operator<< (std::ostream& ostr,eBasicDataType data_type)
{
  ostr << basicDataTypeName(data_type);
  return ostr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" eBasicDataType
basicDataTypeFromName(const char* name,bool& has_error)
{
  has_error = true;
  std::string_view buf(name);
  for( int i=0; i<NB_BASIC_DATA_TYPE; ++i ){
    if (buf==std::string_view(N_ALL_NAMES[i])){
      has_error = false;
      return (eBasicDataType)i;
    }
  }
  return eBasicDataType::Unknown;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Integer
basicDataTypeSize(eBasicDataType type)
{
  Byte b = (Byte)type;
  if (b>=NB_BASIC_DATA_TYPE)
    throw ArgumentException("basicDataTypeSize()","Invalid datatype");
  return ALL_SIZEOF[b];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Trouve le type associé à \a name. Envoie une exception en cas d'erreur
extern "C++" eBasicDataType
basicDataTypeFromName(const char* name)
{
  bool has_error = true;
  eBasicDataType data_type = basicDataTypeFromName(name,has_error);
  if (has_error)
    ARCCORE_FATAL("Bad DataType '{0}'",name);
  return data_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" std::istream&
operator>> (std::istream& istr,eBasicDataType& data_type)
{
  std::string buf;
  istr >> buf;
  bool has_error = true;
  data_type = basicDataTypeFromName(buf.c_str(),has_error);
  if (has_error){
    data_type = eBasicDataType::Unknown;
    istr.setstate(std::ios_base::failbit);
  }
  return istr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

