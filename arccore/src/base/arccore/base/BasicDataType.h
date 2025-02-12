// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicDataType.h                                             (C) 2000-2025 */
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
  Int128 = 9, //!< Donnée de type entier 128 bits
  BFloat16 = 10, //! < Donnée de type BFloat16
  Int8 = 11, //! Donnée de type entier 8 bits
  Real = Float64 //! Donnée de type Float64
};
//! Nombre de types de base supportés
constexpr unsigned char NB_BASIC_DATA_TYPE = 12;

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

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Arccore::eBasicDataType;
using Arccore::basicDataTypeFromName;
using Arccore::basicDataTypeSize;
using Arccore::basicDataTypeName;
using Arccore::NB_BASIC_DATA_TYPE;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
