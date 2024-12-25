// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypes.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Définition des types liées aux données.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DATATYPES_DATATYPES_H
#define ARCANE_CORE_DATATYPES_DATATYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// NOTE: Swig 4.2.0 ne gère pas bien le 'Int32´.
// (cela fonctionne avec Swig 4.1.1)
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type d'une donnée.
 */
#ifndef SWIG
enum eDataType : Int32
#else
enum eDataType
#endif
{
  DT_Byte = 0, //!< Donnée de type octet
  DT_Real = 1, //!< Donnée de type réel
  DT_Int16 = 2, //!< Donnée de type entier 16 bits
  DT_Int32 = 3, //!< Donnée de type entier 32 bits
  DT_Int64 = 4, //!< Donnée de type entier 64 bits
  DT_String = 5, //!< Donnée de type chaîne de caractère UTF-8
  DT_Real2 = 6, //!< Donnée de type vecteur 2
  DT_Real3 = 7, //!< Donnée de type vecteur 3
  DT_Real2x2 = 8, //!< Donnée de type tenseur 3x3
  DT_Real3x3 = 9, //!< Donnée de type tenseur 3x3
  DT_BFloat16 = 10, //!< Donnée de type 'BFloat16'
  DT_Float16 = 11, //!< Donnée de type 'Float16'
  DT_Float32 = 12, //!< Donnée de type 'Float32'
  DT_Int8 = 13, //!< Donnée de type entier sur 8 bits
  DT_Float128 = 14, //!< Donnée de type flottant sur 128 bits
  DT_Int128 = 15, //!< Donnée de type entier sur 128 bits
  DT_Unknown = 16 //!< Donnée de type inconnue ou non initialisée
};

//! Nombre de valeurs de eDataType
static constexpr Int32 NB_ARCANE_DATA_TYPE = 17;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste des noms pour eDataType.
 */
class DataTypeNames
{
 public:
  static constexpr const char* N_BYTE = "Byte";
  static constexpr const char* N_REAL = "Real";
  static constexpr const char* N_INT16 = "Int16";
  static constexpr const char* N_INT32 = "Int32";
  static constexpr const char* N_INT64 = "Int64";
  static constexpr const char* N_STRING = "String";
  static constexpr const char* N_REAL2 = "Real2";
  static constexpr const char* N_REAL3 = "Real3";
  static constexpr const char* N_REAL2x2 = "Real2x2";
  static constexpr const char* N_REAL3x3 = "Real3x3";
  static constexpr const char* N_BFLOAT16 = "BFloat16";
  static constexpr const char* N_FLOAT16 = "Float16";
  static constexpr const char* N_FLOAT32 = "Float32";
  static constexpr const char* N_INT8 = "Int8";
  static constexpr const char* N_FLOAT128 = "Float128";
  static constexpr const char* N_INT128 = "Int128";
  static constexpr const char* N_UNKNOWN = "Unknown";
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Nom du type de donnée.
extern "C++" ARCANE_CORE_EXPORT const char*
dataTypeName(eDataType type);

//! Trouve le type associé à \a name
extern "C++" ARCANE_CORE_EXPORT eDataType
dataTypeFromName(const char* name,bool& has_error);

//! Trouve le type associé à \a name. Envoie une exception en cas d'erreur
extern "C++" ARCANE_CORE_EXPORT eDataType
dataTypeFromName(const char* name);

//! Taille du type de donnée \a type (qui doit être différent de \a DT_String)
extern "C++" ARCANE_CORE_EXPORT Integer
dataTypeSize(eDataType type);

//! Opérateur de sortie sur un flot
extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<< (std::ostream& ostr,eDataType data_type);

//! Opérateur d'entrée depuis un flot
extern "C++" ARCANE_CORE_EXPORT std::istream&
operator>> (std::istream& istr,eDataType& data_type);


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type de politique d'initialisation possible pour une donnée.
 *
 * Par défaut, pour des raisons historiques, il s'agit de DIP_Legacy.
 *
 * La politique d'initialisation est prise en compte pour
 * l'initialisation des variables Arcane. Cette valeur peut être
 * positionnée/récupérée via les fonctions
 * setGlobalDataInitialisationPolicy() et getGlobalDataInitialisationPolicy().
 *
 */
enum eDataInitialisationPolicy
{
  //! Pas d'initialisation forcée
  DIP_None =0,
  /*!
   * \brief Initialisation avec le constructeur par défaut.
   *
   * Pour les entiers, il s'agit de la valeurs 0. Pour les réels, il
   * s'agit de la valeur 0.0.
   */
  DIP_InitWithDefault =1,
  /*!
   * \brief Initialisation avec des NaN (Not a Number)
   *
   * Ce mode permet d'initialiser les données de type Real et dérivés (Real2, Real3, ...)
   * avec des valeurs NaN qui déclenchent une exception lorsqu'elles
   * sont utilisées.
   */
  DIP_InitWithNan = 2,
  /*!
   * \brief Initialisation en mode historique.
   *
   * Ce mode est conservé pour compatibilité avec les versions d'Arcane inférieurs
   * à la version 2.0 où c'était le mode par défaut.
   * Dans ce mode, les variables sur les entités du maillage étaient
   * toujours initialisées avec le constructeur par défaut lors de
   * leur première allocation, quelle que soit la valeur de
   * getGlobalDataInitialisationPolicy(). La politique d'initialisation n'était
   * prise en compte que lors d'un changement du nomnbre d'éléments (resize())
   * où pour les variables qui n'étaient pas des variables du maillage.
   */
  DIP_Legacy = 3,
  /*!
   * \brief Initialisation avec des NaN pour à la création et le constructeur
   * par défaut ensuite.
   *
   * Ce mode est identique à DIP_InitWithNan pour la création des variables
   * et à DIP_InitWithDefault lorsqu'on la taille de la variable évolue
   * (en géneral via un appel à IVariable::resize() ou IVariable::resizeFromGroup()).
   */
  DIP_InitInitialWithNanResizeWithDefault = 4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Positionne la politique d'initialisation des variables.
extern "C++" ARCANE_CORE_EXPORT void 
setGlobalDataInitialisationPolicy(eDataInitialisationPolicy init_policy);

//! Récupère la politique d'initialisation des variables.
extern "C++" ARCANE_CORE_EXPORT eDataInitialisationPolicy
getGlobalDataInitialisationPolicy();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Type de trace possible
enum eTraceType
{
  TT_None = 0,
  TT_Read = 1,
  TT_Write = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

// Macro pour instantier une classe template pour tous les types numériques
#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE1(class_name)\
template class class_name <Real>;\
template class class_name <Real3>;\
template class class_name <Real3x3>;\
template class class_name <Real2>;\
template class class_name <Real2x2>;

// Macro pour instantier une classe template pour tous les types numériques
#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE2(class_name)\
template class class_name <Int8>;\
template class class_name <Int16>;\
template class class_name <Int32>;\
template class class_name <Int64>;\
template class class_name <Byte>

// Macro pour instantier une classe template pour tous les types numériques
#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE3(class_name)\
template class class_name <BFloat16>;\
template class class_name <Float16>;\
template class class_name <Float32>;

// Macro pour instantier une classe template pour tous les types numériques
#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(class_name)\
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE1(class_name);\
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE2(class_name);\
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE3(class_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
