// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicDataType.h                                             (C) 2000-2019 */
/*                                                                           */
/* Définition des types de données basiques.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BASICDATATYPE_H
#define ARCCORE_BASE_BASICDATATYPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type d'une donnée de base.
 *
 * \note: les valeurs doivent être contigues et tenir sur un octet et ne
 * doivent pas être modifiées car elles sont utilisés pour la sérialisation
 * par exemple. Il est cependant possible d'en ajouter.
 */
enum class eBasicDataType : unsigned char
{
  Unknown = 0, //!< Donnée de type inconnu ou non initialisé
  Byte = 1, //!< Donnée de type octet
  Float16 = 2, //!< Donnée de type Float16
  Float32 = 3, //!< Donnée de type Float32
  Float64 = 4, //!< Donnée de type Float64
  Float128 = 5, //!< Donnée de type Float128
  Int16 = 6, //!< Donnée de type entier 16 bits
  Int32 = 7, //!< Donnée de type entier 32 bits
  Int64 = 8, //!< Donnée de type entier 64 bits
  Int128 = 9 //!< Donnée de type entier 128 bits
};
//! Nombre de types de base supportés
constexpr unsigned char NB_BASIC_DATA_TYPE = 10;

//! Nom du type de donnée.
extern "C++" ARCCORE_BASE_EXPORT const char*
basicDataTypeName(eBasicDataType type);

//! Trouve le type associé à \a name
extern "C++" ARCCORE_BASE_EXPORT eBasicDataType
basicDataTypeFromName(const char* name,bool& has_error);

//! Trouve le type associé à \a name. Envoie une exception en cas d'erreur
extern "C++" ARCCORE_BASE_EXPORT eBasicDataType
basicDataTypeFromName(const char* name);

//! Taille du type de donnée \a type
extern "C++" ARCCORE_BASE_EXPORT Integer
basicDataTypeSize(eBasicDataType type);

//! Opérateur de sortie sur un flot
extern "C++" ARCCORE_BASE_EXPORT std::ostream&
operator<< (std::ostream& ostr,eBasicDataType data_type);

//! Opérateur d'entrée depuis un flot
extern "C++" ARCCORE_BASE_EXPORT std::istream&
operator>> (std::istream& istr,eBasicDataType& data_type);


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
