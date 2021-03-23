// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypes.h                                                 (C) 2000-2018 */
/*                                                                           */
/* Définition des types liées aux données.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPES_DATATYPES_H
#define ARCANE_DATATYPES_DATATYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Type d'une donnée.
 *
 */
enum eDataType
{
  // NOTE: ne pas changer ces valeurs car elles sont utilisées dans Arccore
  DT_Byte = 0, //!< Donnée de type octet
  DT_Real, //!< Donnée de type réel
  DT_Int16, //!< Donnée de type entier 16 bits
  DT_Int32, //!< Donnée de type entier 32 bits
  DT_Int64, //!< Donnée de type entier 64 bits
  DT_String, //!< Donnée de type chaîne de caractère unicode
  DT_Real2, //!< Donnée de type vecteur 2
  DT_Real3, //!< Donnée de type vecteur 3
  DT_Real2x2, //!< Donnée de type tenseur 3x3
  DT_Real3x3, //!< Donnée de type tenseur 3x3
  DT_Unknown  //!< Donnée de type inconnu ou non initilialisé
};

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

/*!
 * \brief Allocateur par défaut pour les données.
 *
 * Cette allocateur utilise celui platform::getAcceleratorHostMemoryAllocator()
 * s'il est disponible, sinon il utilise un allocateur aligné.
 *
 * Il est garanti que l'allocateur retourné permettra d'utiliser la donnée
 * sur accélerateur si cela est disponible.
 *
 * Il est garanti que l'alignement est au moins celui retourné par
 * AlignedMemoryAllocator::Simd().
 */
extern "C++" ARCANE_CORE_EXPORT IMemoryAllocator*
arcaneDefaultDataAllocator();

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
  DIP_Legacy = 3
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
