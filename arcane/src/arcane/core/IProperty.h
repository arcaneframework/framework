// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IProperty.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Property interface.                                                       */
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
 * \brief Definition of types for properties.
 */
class Property
{
 public:

  virtual ~Property() = default;

 public:

  //! Kind of a property
  enum ePropertyKind
  {
    PkSimple, //!< Simple kind (Real, integer, string, ...)
    PkEnum, //!< Enumerated kind
    PkExtended, //!< Extended kind
    PkComplex //!< Complex kind containing sub-types
  };
  //! Simple type in the case of a PkSimple kind
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
 * \brief Interface of a property value.
 */
class IPropertyValue
: public Property
{
 public:

  //! Stores the property value in \a str
  virtual void valueToString(String& str) const = 0;
  //! Stores the property name in \a str
  virtual void nameToString(String& str) const = 0;
  //! Positions the new property value at \a str
  virtual void setValueFromString(const String& str) = 0;
  //! Returns whether the current value is the default value
  virtual bool isDefaultValue() const = 0;
  //! Returns whether the current value is the original value
  virtual bool isOriginalValue() const = 0;
  //! Stores the original property value in \a str
  virtual void originalValueToString(String& str) const = 0;
  //! Returns whether the value can be edited.
  virtual bool canBeEdited() const = 0;
  //! Returns the type of this property.
  virtual IPropertyType* type() = 0;
  //! Returns the instance associated with this value.
  virtual IPropertyTypeInstance* typeInstance() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a property type.
 */
class IPropertyType
: public Property
{
 public:

  //! Returns the name of the type in \a str.
  virtual void typeNameToString(String& str) const = 0;

  //! Returns the kind of the property type
  virtual ePropertyKind kind() const = 0;

  /*!
   * \brief Returns the number of possible enumerated values for the type.
   * This method is only useful for PkEnum kind types. In
   * other cases, it returns zero.
   */
  virtual Integer nbEnumeratedValue() const = 0;

  /*!
   * \brief Returns the i-th enumerated value of the type.
   * This method is only useful for PkEnum kind types. In
   * other cases, it returns the null string.
   */
  virtual String enumeratedValue(Integer i) const = 0;

  /*!
   * \brief Returns the simple type of the property storage.
   * This method is only valid for PkSimple kind types. For all
   * other types, it returns StString.
   */
  virtual eSimpleType simpleType() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a property type instance.
 */
class IPropertyTypeInstance
: public Property
{
 public:

  //! Stores the instance name in \a str
  virtual void nameToString(String& str) const = 0;
  //! Returns the type of the instance.
  virtual IPropertyType* type() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
