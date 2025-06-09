// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IProperty.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Interface des propriétés.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPROPERTY_H
#define ARCANE_CORE_IPROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IPropertyValue;
class IPropertyType;
class IPropertyTypeInstance;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Définition des types pour les propriétés.
 */
class Property
{
 public:

  virtual ~Property() = default;

 public:

  //! Genre d'une propriété
  enum ePropertyKind
  {
    PkSimple, //!< Genre simple (Réel, entier, chaîne, ...)
    PkEnum, //!< Genre énuméré
    PkExtended, //!< Genre étendu
    PkComplex //!< Genre complexe contenant des sous-types
  };
  //! Type simple dans le cas d'un genre PkSimple
  enum eSimpleType
  {
    StString,
    StReal,
    StInteger,
    StBool
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une valeur propriété.
 */
class IPropertyValue
: public Property
{
 public:

  //! Stocke la valeur de la propriété dans \a str
  virtual void valueToString(String& str) const = 0;
  //! Stocke le nom de la propriété dans \a str
  virtual void nameToString(String& str) const = 0;
  //! Positionne la nouvelle valeur de la propriété à \a str
  virtual void setValueFromString(const String& str) = 0;
  //! Retourne si la valeur actuelle est la valeur par défaut
  virtual bool isDefaultValue() const = 0;
  //! Retourne si la valeur actuelle est la valeur originelle
  virtual bool isOriginalValue() const = 0;
  //! Stocke la valeur originale de la propriété dans \a str
  virtual void originalValueToString(String& str) const = 0;
  //! Retourne si la valeur peut être éditée.
  virtual bool canBeEdited() const = 0;
  //! Retourne le type de cette propriété.
  virtual IPropertyType* type() = 0;
  //! Retourne l'instance associé à cette valeur.
  virtual IPropertyTypeInstance* typeInstance() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un type de propriété.
 */
class IPropertyType
: public Property
{
 public:

  //! Retourne dans \a str le nom du type.
  virtual void typeNameToString(String& str) const = 0;

  //! Retourne le genre du type de la propriété
  virtual ePropertyKind kind() const = 0;

  /*!
   * \brief Retourne le nombre de valeurs enumérées possible pour le type.
   * Cette méthode n'est utile que les types du genre PkEnum. Dans les
   * autres cas, elle retourne zéro.
   */
  virtual Integer nbEnumeratedValue() const = 0;

  /*!
   * \brief Retourne la \a ième valeur enumérée du type.
   * Cette méthode n'est utile que les types du genre PkEnum. Dans les
   * autres cas, elle retourne la chaîne nulle.
   */
  virtual String enumeratedValue(Integer i) const = 0;

  /*!
   * \brief Retourne le type simple du stockage de la propriété.
   * Cette méthode n'est valide que les types du genre PkSimple. Pour tous les
   * autres types, elle retourne StString.
   */
  virtual eSimpleType simpleType() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une instance d'un type de propriété.
 */
class IPropertyTypeInstance
: public Property
{
 public:

  //! Stocke le nom de l'instance dans \a str
  virtual void nameToString(String& str) const = 0;
  //! Retourne le type de l'instance.
  virtual IPropertyType* type() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
