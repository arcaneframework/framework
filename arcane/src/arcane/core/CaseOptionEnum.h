// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionEnum.h                                            (C) 2000-2023 */
/*                                                                           */
/* Enumerated data set option.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONENUM_H
#define ARCANE_CASEOPTIONENUM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionSimple.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Name of a data set option.
 * This class allows storing the name of an option in multiple languages.
 */
class ARCANE_CORE_EXPORT CaseOptionName
{
 public:

  //! Constructs a name option \a true_name
  CaseOptionName(const String& true_name);
  //! Copy constructor
  CaseOptionName(const CaseOptionName& rhs);
  //! Releases resources
  virtual ~CaseOptionName();

 public:

  /*! \brief returns the name of the option in the language \a lang.
   * If no translation is available in the language \a lang,
   * trueName() is returned.
   */
  String name(const String& lang) const;
  //! Returns the true name (non-translated) of the option
  String trueName() const { return m_true_name; }
  /*!
    \brief Adds a translation for the option name.
    Adds the name \a tname corresponding to the language \a lang.
    If a translation already exists for this language, it is replaced by
    this one.
    \param tname translation of the name
    \param lang language of the translation
  */
  void addAlternativeNodeName(const String& lang, const String& tname);

 private:

  String m_true_name; //!< Option name
  StringDictionary* m_translations; //!< Translations.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Name and value of a data set enumeration.
 */
class ARCANE_CORE_EXPORT CaseOptionEnumValue
: public CaseOptionName
{
 public:

  CaseOptionEnumValue(const String& name, int value);
  //! Copy constructor
  CaseOptionEnumValue(const CaseOptionEnumValue& rhs);
  ~CaseOptionEnumValue();

 public:

  int value() const { return m_value; }

 private:

  int m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Set of values for an enumeration.
 */
class ARCANE_CORE_EXPORT CaseOptionEnumValues
{
 public:

  //! Type of the value list
  typedef UniqueArray<CaseOptionEnumValue*> EnumValueList;

 public:

  //! Constructs the instance
  CaseOptionEnumValues();
  ~CaseOptionEnumValues(); //!< Releases resources

 public:

  /*! \brief Adds the enumeration value \a value.
   * The instance becomes the owner of \a value, which is destroyed
   * when it is no longer used.
   * This function should only be called during initialization.
   * If \a do_clone is true, a copy of \a value is used
   */
  void addEnumValue(CaseOptionEnumValue* value, bool do_clone);

  //! Returns the number of enumeration values
  Integer nbEnumValue() const;

  //! Returns the i-th value
  CaseOptionEnumValue* enumValue(Integer index) const;

  /*! \brief Returns the value of the enumeration having the name \a name
   *
   * The value is returned in \a index.
   * \param name name of the enumeration
   * \param lang is the language of the data set
   * \param value is the enumeration value (returned)
   * \retval true in case of error,
   * \retval false in case of success.
   */
  bool valueOfName(const String& name, const String& lang, int& value) const;

  //! Returns the name corresponding to the value \a value for the language \a lang
  String nameOfValue(int value, const String& lang) const;

  /*!
   * \brief Fills \a names with valid names for the language \a lang.
   */
  void getValidNames(const String& lang, StringArray& names) const;

 private:

  EnumValueList* m_enum_values; //!< Enumeration values
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Enumerated data set option.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionEnum
: public CaseOptionSimple
{
 public:

  CaseOptionEnum(const CaseOptionBuildInfo& cob, const String& type_name);
  ~CaseOptionEnum();

 public:

  virtual void print(const String& lang, std::ostream& o) const;
  virtual void updateFromFunction(Real current_time, Integer current_iteration)
  {
    _updateFromFunction(current_time, current_iteration);
  }

  void addEnumValue(CaseOptionEnumValue* value, bool do_clone)
  {
    m_enum_values->addEnumValue(value, do_clone);
  }
  CaseOptionEnumValues* enumValues() const { return m_enum_values; }

  virtual void visit(ICaseDocumentVisitor* visitor) const;

  int enumValueAsInt() const { return _optionValue(); }

 public:
 protected:

  virtual void _search(bool is_phase1);
  virtual bool _allowPhysicalUnit() { return false; }

  //! Sets the option value to \a v
  virtual void _setOptionValue(int v) = 0;
  //! Returns the option value
  virtual int _optionValue() const = 0;

 protected:

  void _setEnumDefaultValue(int def_value);

 private:

  String m_type_name; //!< Enumeration name
  CaseOptionEnumValues* m_enum_values;
  void _updateFromFunction(Real current_time, Integer current_iteration);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup CaseOption
 * \brief Enumerated data set option.
 *
 * \a T is the computer type of the enumeration.
 */
template <class EnumType>
class CaseOptionEnumT
: public CaseOptionEnum
{
 public:

  CaseOptionEnumT(const CaseOptionBuildInfo& cob, const String& type_name)
  : CaseOptionEnum(cob, type_name)
  , m_value(EnumType())
  {}

 public:

  //! Option value
  EnumType value() const
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return m_value;
  }

  //! Option value
  operator EnumType() const { return value(); }

  //! Option value
  EnumType operator()() const { return value(); }

  /*!
   * \brief Sets the default value of the option.
   *
   * If the option is not present in the data set, its value will be
   * that specified by the argument \a def_value; otherwise, calling this method has no effect.
   */
  void setDefaultValue(EnumType def_value)
  {
    _setEnumDefaultValue(static_cast<int>(def_value));
  }

  //! Returns the option value if isPresent()==true, otherwise \a arg_value
  EnumType valueIfPresentOrArgument(EnumType arg_value)
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return isPresent() ? m_value : arg_value;
  }

 private:

  EnumType m_value; //!< Option value

 public:
 protected:

  virtual void _setOptionValue(int i)
  {
    m_value = static_cast<EnumType>(i);
  }
  virtual int _optionValue() const
  {
    return static_cast<int>(m_value);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Multi-enumeration data set option.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionMultiEnum
: public CaseOptionBase
{
 public:
 public:

  CaseOptionMultiEnum(const CaseOptionBuildInfo& cob, const String& type_name);
  ~CaseOptionMultiEnum();

 public:

  virtual void print(const String& lang, std::ostream& o) const;
  virtual ICaseFunction* function() const { return 0; }
  virtual void updateFromFunction(Real /*current_time*/, Integer /*current_iteration*/) {}

  void addEnumValue(CaseOptionEnumValue* value, bool do_clone)
  {
    m_enum_values->addEnumValue(value, do_clone);
  }

  CaseOptionEnumValues* enumValues() const { return m_enum_values; }

  virtual void visit(ICaseDocumentVisitor* visitor) const;

 protected:

  virtual void _search(bool is_phase1);
  virtual bool _allowPhysicalUnit() { return false; }

  //! Allocates an array for \a size elements
  virtual void _allocate(Integer size) = 0;
  //! Returns the number of elements in the array.
  virtual Integer _nbElem() const = 0;
  /*! Sets the option value to the value \a v.
   * \a v is directly converted to the enumeration value.
   */
  virtual void _setOptionValue(Integer index, int v) = 0;
  //! Returns the enumeration value for index \a index.
  virtual int _optionValue(Integer index) const = 0;

 private:

  String m_type_name; //!< Enumeration name
  CaseOptionEnumValues* m_enum_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Multi-enumeration data set option.
 * \ingroup CaseOption
 */
template <class T>
class CaseOptionMultiEnumT
: public CaseOptionMultiEnum
, public ArrayView<T>
{
 public:

  typedef T Type; //!< Option type.

 public:

  CaseOptionMultiEnumT(const CaseOptionBuildInfo& cob, const String type_name)
  : CaseOptionMultiEnum(cob, type_name)
  {}

 protected:

  virtual void _allocate(Integer size)
  {
    m_values.resize(size);
    ArrayView<T>* view = this;
    *view = m_values.view();
  }
  virtual Integer _nbElem() const
  {
    return this->size();
  }
  virtual void _setOptionValue(Integer index, int v)
  {
    (*this)[index] = static_cast<T>(v);
  }
  virtual int _optionValue(Integer index) const
  {
    return static_cast<int>((*this)[index]);
  }

 private:

  UniqueArray<T> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
