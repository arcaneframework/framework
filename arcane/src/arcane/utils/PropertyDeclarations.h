﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PropertyDeclarations.h                                      (C) 2000-2020 */
/*                                                                           */
/* Déclaration des types et macros pour la gestion des propriétés.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PROPERTYDECLARATIONS_H
#define ARCANE_UTILS_PROPERTYDECLARATIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * NOTE: Les classes de ce fichier sont en cours de mise au point.
 * NOTE: L'API peut changer à tout moment. Ne pas utiliser en dehors de Arcane.
 */

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class JSONValue;
}

namespace Arcane::properties
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IPropertyVisitor;
template<typename T>
class PropertyVisitor;
template<typename T>
class GenericPropertyVisitorWrapper;
template<typename T>
class PropertyDeclaration
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour déclarer les informations de propriété dans une classe
 *
 * Cette macro doit s'utiliser dans la définition d'une classe. L'argument
 * de la macro doit être le nom de la classe. Par exemple:
 *
 * \code
 * class MyClass
 * {
 *  public:
 *   ARCANE_DECLARE_PROPERTY_CLASS(MyClass);
 * };
 * \endcode
 */
#define ARCANE_DECLARE_PROPERTY_CLASS(class_name)\
 public:\
  static const char* propertyClassName() { return #class_name; }\
  template<typename V> static void _applyPropertyVisitor(V& visitor);\
  static void applyPropertyVisitor(Arcane::properties::PropertyVisitor<class_name>& p); \
  static void applyPropertyVisitor(Arcane::properties::IPropertyVisitor* p)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour enregistrer une classe contenant des propriétés.
 *
 * La classe spécifiée doit contenir la macro ARCANE_PROPERTY_CLASS_DECLARE.
 * La classe doit aussi définir une méthode template _applyPropertyVisitor().
 *
 * Par exemple:
 * \code
 * // Header
 * class MyClass
 * {
 *   ARCANE_DECLARE_PROPERTY_CLASS(MyClass);
 * };
 *
 * // Source code
 * template<typename V> void MyClass::
 * _applyPropertyVisitor(V& p)
 * {
 * }
 *
 * ARCANE_REGISTER_PROPERTY_CLASS(MyClass,());
 * \endcode
 */
#define ARCANE_REGISTER_PROPERTY_CLASS(aclass,a_build_args) \
namespace\
{\
  Arcane::properties::IPropertySettingsInfo*                                     \
  ARCANE_JOIN_WITH_LINE(arcaneCreatePropertySettingsInfo##aclass) (const Arcane::properties::PropertySettingsBuildInfo& sbi) \
  {\
    auto* si = Arcane::properties::PropertySettingsInfo<aclass>::create(sbi,__FILE__,__LINE__); \
    return si;\
  }\
  Arcane::properties::PropertySettingsBuildInfo \
  ARCANE_JOIN_WITH_LINE(arcaneCreatePropertySettingsBuildInfo##aclass) () \
  {\
    return Arcane::properties::PropertySettingsBuildInfo a_build_args;\
  }\
}\
void aclass :: \
applyPropertyVisitor(Arcane::properties::PropertyVisitor<aclass>& p)\
{\
  aclass :: _applyPropertyVisitor(p);\
}\
void aclass :: \
applyPropertyVisitor(Arcane::properties::IPropertyVisitor* p) \
{\
  Arcane::properties::GenericPropertyVisitorWrapper<aclass> xp(p);\
  aclass :: _applyPropertyVisitor(xp); \
}\
Arcane::properties::PropertySettingsRegisterer ARCANE_EXPORT \
 ARCANE_JOIN_WITH_LINE(globalPropertySettingsRegisterer##aclass)\
  (& ARCANE_JOIN_WITH_LINE(arcaneCreatePropertySettingsInfo##aclass),\
   & ARCANE_JOIN_WITH_LINE(arcaneCreatePropertySettingsBuildInfo##aclass),\
   #aclass)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::properties

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

