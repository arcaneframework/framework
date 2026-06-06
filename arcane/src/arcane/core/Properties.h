// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Properties.h                                                (C) 2000-2025 */
/*                                                                           */
/* List of properties.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PROPERTIES_H
#define ARCANE_CORE_PROPERTIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/AutoRef.h"
#include "arcane/core/SharedReference.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PropertiesImpl;
class IPropertyMng;

class PropertiesImplBase
: public SharedReference
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief List of properties.
 *
 * This class manages a list of properties. A property is
 * characterized by a name and a value of a given type.
 * The name must not contain the character '.' which serves as a delimiter
 * for property hierarchies.
 *
 * The set*() functions allow positioning a property. The
 * get*() functions allow retrieving the value.
 *
 * For scalar properties, there are three ways to retrieve a
 * value. These three methods are equivalent unless the property has not been set.
 * - via an overload of the get() method. If the property has not been set,
 * the passed argument value is unchanged and the method returns false.
 * - via an explicit call (e.g., getBool()). If the property has not been set,
 * the value obtained with the default constructor for the relevant type is used.
 * - via an explicit call with a possible default value (e.g., getBoolWithDefault()).
 * If the property has not been set, the default value passed as an argument is
 * used.
 *
 */
class ARCANE_CORE_EXPORT Properties
{
 public:

  //! Creates or retrieves a list of properties with name \a name
  Properties(IPropertyMng* pm, const String& name);

  //! Creates or retrieves a list of properties with name \a name and child of \a parent_property
  Properties(const Properties& parent_property, const String& name);

  //! Copy constructor
  Properties(const Properties& rhs);
  //! Copy assignment operator
  const Properties& operator=(const Properties& rhs);
  //! Destroys the reference to this property
  virtual ~Properties();

 public:

  //! Sets a boolean property of name \a name and value \a value.
  void setBool(const String& name, bool value);

  //! Sets a boolean property of name \a name and value \a value.
  void set(const String& name, bool value);

  //! Value of the property named \a name.
  bool getBool(const String& name) const;

  //! Value of the property named \a name.
  bool getBoolWithDefault(const String& name, bool default_value) const;

  //! Value of the property named \a name.
  bool get(const String& name, bool& value) const;

  //! Sets an Int32 property of name \a name and value \a value.
  void setInt32(const String& name, Int32 value);

  //! Sets an Int32 property of name \a name and value \a value.
  void set(const String& name, Int32 value);

  //! Value of the property named \a name.
  Int32 getInt32(const String& name) const;

  //! Value of the property named \a name.
  Int32 getInt32WithDefault(const String& name, Int32 default_value) const;

  //! Value of the property named \a name.
  bool get(const String& name, Int32& value) const;

  //! Sets an Int64 property of name \a name and value \a value.
  void setInt64(const String& name, Int64 value);

  //! Sets an Int64 property of name \a name and value \a value.
  void set(const String& name, Int64 value);

  //! Value of the property named \a name.
  Int64 getInt64(const String& name) const;

  //! Value of the property named \a name.
  Int64 getInt64WithDefault(const String& name, Int64 default_value) const;

  //! Value of the property named \a name.
  bool get(const String& name, Int64& value) const;

  //! Sets an Integer property of name \a name and value \a value.
  void setInteger(const String& name, Integer value);

  //! Value of the property named \a name.
  Integer getInteger(const String& name) const;

  //! Value of the property named \a name.
  Integer getIntegerWithDefault(const String& name, Integer default_value) const;

  //! Sets a Real property of name \a name and value \a value.
  void setReal(const String& name, Real value);

  //! Sets a Real property of name \a name and value \a value.
  void set(const String& name, Real value);

  //! Value of the property named \a name.
  Real getReal(const String& name) const;

  //! Value of the property named \a name.
  Real getRealWithDefault(const String& name, Real default_value) const;

  //! Value of the property named \a name.
  bool get(const String& name, Real& value) const;

  //! Sets a String property of name \a name and value \a value.
  void setString(const String& name, const String& value);

  //! Sets a String property of name \a name and value \a value.
  void set(const String& name, const String& value);

  //! Value of the property named \a name.
  String getString(const String& name) const;

  //! Value of the property named \a name.
  String getStringWithDefault(const String& name, const String& default_value) const;

  //! Value of the property named \a name.
  bool get(const String& name, String& value) const;

  //! Sets a BoolUniqueArray property of name \a name and value \a value.
  void set(const String& name, BoolConstArrayView value);

  //! Value of the property named \a name.
  void get(const String& name, BoolArray& value) const;

  //! Sets an Int32UniqueArray property of name \a name and value \a value.
  void set(const String& name, Int32ConstArrayView value);

  //! Value of the property named \a name.
  void get(const String& name, Int32Array& value) const;

  //! Sets an Int64UniqueArray property of name \a name and value \a value.
  void set(const String& name, Int64ConstArrayView value);

  //! Value of the property named \a name.
  void get(const String& name, Int64Array& value) const;

  //! Sets a RealUniqueArray property of name \a name and value \a value.
  void set(const String& name, RealConstArrayView value);

  //! Value of the property named \a name.
  void get(const String& name, RealArray& value) const;

  //! Sets a StringUniqueArray property of name \a name and value \a value.
  void set(const String& name, StringConstArrayView value);

  //! Value of the property named \a name.
  void get(const String& name, StringArray& value) const;

 public:

  //! Prints the properties and their values to the stream \a o
  void print(std::ostream& o) const;

  //! Performs the serialization of the properties
  void serialize(ISerializer* serializer);

  //! Name of the property.
  const String& name() const;

  //! Full name of the property.
  const String& fullName() const;

  IPropertyMng* propertyMng() const;

  /*!
   * \brief Destroys the associated values of properties linked to this reference.
   */
  void destroy();

  //! \internal
  PropertiesImpl* impl() const { return m_p; }

  //! \internal
  PropertiesImplBase* baseImpl() const { return m_ref.get(); }

 private:

  PropertiesImpl* m_p;
  AutoRefT<PropertiesImplBase> m_ref;

 private:

  Properties(PropertiesImpl* p);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
