// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionSimple.h                                          (C) 2000-2025 */
/*                                                                           */
/* Simple data set option.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONSIMPLE_H
#define ARCANE_CASEOPTIONSIMPLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ICaseOptions.h"
#include "arcane/core/CaseOptionBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class IStandardFunction;
class IPhysicalUnitConverter;
class StringDictionary;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_CHECK
#define ARCANE_CASEOPTION_CHECK_IS_INITIALIZED _checkIsInitialized()
#else
#define ARCANE_CASEOPTION_CHECK_IS_INITIALIZED
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Type>
class CaseOptionTraitsT
{
 public:

  using ContainerType = Type;
  using ReferenceType = Type&;
  using ConstReferenceType = const Type&;
  using ArrayViewType = ArrayView<Type>;
  using ConstArrayViewType = ConstArrayView<Type>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Specialization for 'Array' options.
 *
 * This is necessary because the 'Array' class cannot be instantiated. For
 * this type of option, it is forbidden to modify the option values, so
 * the views must necessarily be constant.
 */
template <typename Type>
class CaseOptionTraitsT<Array<Type>>
{
 public:

  using ContainerType = UniqueArray<Type>;
  using ReferenceType = const Array<Type>&;
  using ConstReferenceType = const Array<Type>&;
  using ArrayViewType = ConstArrayView<ContainerType>;
  using ConstArrayViewType = ConstArrayView<ContainerType>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for simple options (single value).
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionSimple
: public CaseOptionBase
{
 public:

  explicit CaseOptionSimple(const CaseOptionBuildInfo& cob);
  CaseOptionSimple(const CaseOptionBuildInfo& cob, const String& physical_unit);
  ~CaseOptionSimple();

 public:

  //! Returns \a true if the option is present
  bool isPresent() const { return !m_element.null(); }

  /*!
   * \brief Returns the element of the option.
   *
   * \deprecated The internal implementation should not be used to allow
   * for the use of a format other than XML in the future.
   */
  ARCANE_DEPRECATED_LONG_TERM("Y2022: Do not access XML item from option")
  XmlNode element() const { return m_element; }

  /*!
   * \brief Function associated with this option (0 if none).
   *
   * If a function is associated with the option, the values of this
   * latter are recalculated automatically at each iteration.
   */
  ICaseFunction* function() const override { return m_function; }
  /*!
   * \brief Standard function associated with this option (0 if none).
   *
   * A standard function has a specific prototype and can be called
   * directly. Unlike function(), the presence of a standard function
   * does not change the value of the option.
   */
  virtual IStandardFunction* standardFunction() const { return m_standard_function; }
  /*!
   * \brief Indicates if the value has changed since the last iteration.
   *
   * The value can only change if a function is associated with the option.
   * The method returns true if the option's value is different from the
   * previous iteration. This method also works in case of rollback.
   */
  bool hasChangedSinceLastIteration() const;
  //! Full name in the format provided by the XPath standard.
  String xpathFullName() const;
  /*!
   * \brief Default physical unit for this option (null if no unit),
   * specified in the .axl file.
   */
  String defaultPhysicalUnit() const;
  //! Physical unit specified in the data set (null if no unit)
  String physicalUnit() const;
  /*!
   * \brief Physical unit converter.
   *
   * This converter only exists for 'Real' or 'RealArray' type options.
   * It is null if the option does not have a unit.
   */
  IPhysicalUnitConverter* physicalUnitConverter() const { return m_unit_converter; }
  /*!
   * \brief Indicates if the option is optional.
   *
   * If an optional option is not provided,
   * its value is undefined and should therefore not be used.
   */
  bool isOptional() const { return m_is_optional; }

  /*!
   * \brief Indicates if the option has an invalid value.
   *
   * This is always the case, unless the option is optional
   * (isOptional()==true) and not provided.
   */
  bool hasValidValue() const { return m_has_valid_value; }

  void visit(ICaseDocumentVisitor* visitor) const override;

 protected:

  void _search(bool is_phase1) override;
  virtual bool _allowPhysicalUnit() = 0;
  void _setChangedSinceLastIteration(bool has_changed);
  void _searchFunction(XmlNode& velem);
  void _setPhysicalUnit(const String& value);
  void _setHasValidValue(bool v) { m_has_valid_value = v; }
  XmlNode _element() const { return m_element; }

  static String _convertFunctionRealToString(ICaseFunction* func, Real t);
  static String _convertFunctionIntegerToString(ICaseFunction* func, Integer t);

 private:

  XmlNode m_element; //!< Option element
  ICaseFunction* m_function = nullptr; //!< Associated function (or nullptr)
  IStandardFunction* m_standard_function = nullptr; //!< Associated standard function (or nullpt)
  //! Unit converter (nullptr if not needed). Valid only for 'Real' options
  IPhysicalUnitConverter* m_unit_converter = nullptr;
  bool m_changed_since_last_iteration = false;
  bool m_is_optional = false;
  bool m_has_valid_value = false;
  String m_default_physical_unit;
  String m_physical_unit;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup CaseOption
 * \brief Simple data set option (real, integer, boolean, ...).
 *
 * The most used method of this class is the operator()()
 * which allows retrieving the option's value. If a function (ICaseFunction)
 * is associated with the option, it is possible to retrieve the option's value
 * at the physical time or at the iteration passed as a parameter to the method
 * valueAtParameter().
 * \code
 * CaseOptionSimpleT<Real> real_option;
 * Real v = real_option(); // uses operator()
 * Real v = real_option; // uses implicit cast operator
 * Real v = real_option.valueAtParameter(0.3); // value at physical time 0.3
 * \endcode
 */
template <class T>
class CaseOptionSimpleT
: public CaseOptionSimple
{
 public:

  typedef CaseOptionSimpleT<T> ThatClass;
#ifndef SWIG
  typedef typename CaseOptionTraitsT<T>::ContainerType Type; //!< Option type
#else
  typedef T Type; //!< Option type
#endif
 public:

  ARCANE_CORE_EXPORT CaseOptionSimpleT(const CaseOptionBuildInfo& cob);
  ARCANE_CORE_EXPORT CaseOptionSimpleT(const CaseOptionBuildInfo& cob, const String& physical_unit);

 public:

  ARCANE_CORE_EXPORT virtual void print(const String& lang, std::ostream& o) const;

  //! Returns the value of the option
  const Type& value() const
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return m_value;
  }

  //! Value of the option
  operator const Type&() const { return value(); }

  //! Returns the value of the option for the real parameter t.
  ARCANE_CORE_EXPORT Type valueAtParameter(Real t) const;

  //! Returns the value of the option for the integer parameter t.
  ARCANE_CORE_EXPORT Type valueAtParameter(Integer t) const;

  //! Returns the value of the option
  //const Type& operator()() const { return value(); }
  const Type& operator()() const { return value(); }

#ifdef ARCANE_DOTNET
  operator ThatClass*() { return this; }
  operator const ThatClass*() const { return this; }
  const ThatClass* operator->() const { return this; }
  static Real castTo__Arcane_Real(const ThatClass& v)
  {
    return (Real)(v);
  }
#endif

  //! Returns the value of the option for the real parameter t.
  ARCANE_DEPRECATED Type operator()(Real t) const
  {
    return valueAtParameter(t);
  }

  //! Returns the value of the option for the integer parameter t.
  ARCANE_DEPRECATED Type operator()(Integer t) const
  {
    return valueAtParameter(t);
  }

  /*!
   * For internal use only
   * \internal
   */
  ARCANE_CORE_EXPORT virtual void updateFromFunction(Real current_time, Integer current_iteration);

  /*!
   * \brief Sets the default value of the option.
   *
   * If the option is not present in the data set, its value will be
   * that specified by the argument \a def_value; otherwise, calling this
   * method has no effect.
   */
  ARCANE_CORE_EXPORT void setDefaultValue(const Type& def_value);

  //! Returns the value of the option if isPresent()==true or otherwise \a arg_value
  const Type& valueIfPresentOrArgument(const Type& arg_value)
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return isPresent() ? m_value : arg_value;
  }

 protected:

  ARCANE_CORE_EXPORT virtual void _search(bool is_phase1);
  ARCANE_CORE_EXPORT virtual bool _allowPhysicalUnit();

 private:

  Type m_value; //!< Option value
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT CaseOptionMultiSimple
: public CaseOptionBase
{
 public:

  CaseOptionMultiSimple(const CaseOptionBuildInfo& cob)
  : CaseOptionBase(cob)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup CaseOption
 * \brief Data set option of simple type list (real, integer, boolean, ...).
 *
 * \warning Using the base class `ArrayView<T>` is obsolete and
 * should no longer be used. The view() method allows retrieving a view on the
 * option values.
 */
#ifndef SWIG
template <class T>
class CaseOptionMultiSimpleT
: public CaseOptionMultiSimple
#ifdef ARCANE_HAS_PRIVATE_CASEOPTIONSMULTISIMPLE_BASE_CLASS
, private ArrayView<T>
#else
, public ArrayView<T>
#endif
{
 public:

  //! Type of the option value
  using Type = typename CaseOptionTraitsT<T>::ContainerType;
  using ReferenceType = typename CaseOptionTraitsT<T>::ReferenceType;
  using ConstReferenceType = typename CaseOptionTraitsT<T>::ConstReferenceType;
  //! Type of the view on the option values
  using ArrayViewType = typename CaseOptionTraitsT<T>::ArrayViewType;
  //! Type of the constant view on the option values
  using ConstArrayViewType = typename CaseOptionTraitsT<T>::ConstArrayViewType;

 public:

  ARCANE_CORE_EXPORT CaseOptionMultiSimpleT(const CaseOptionBuildInfo& cob);
  ARCANE_CORE_EXPORT CaseOptionMultiSimpleT(const CaseOptionBuildInfo& cob, const String& physical_unit);
  ARCANE_CORE_EXPORT ~CaseOptionMultiSimpleT();

 public:

  ARCCORE_DEPRECATED_2021("Use view() instead")
  ArrayView<T> operator()()
  {
    return *this;
  }
  ARCCORE_DEPRECATED_2021("Use view() instead")
  const ArrayView<T> operator()() const
  {
    return *this;
  }

  //! Conversion to the constant view
  ARCCORE_DEPRECATED_2021("Use view() instead")
  operator ArrayView<T>()
  {
    ArrayView<T>* v = this;
    return *v;
  }

  //! Conversion to the constant view
  ARCCORE_DEPRECATED_2021("Use view() instead")
  operator ConstArrayView<T>() const
  {
    const ArrayView<T>* v = this;
    return *v;
  }

  //! Constant view on the option elements
  ConstArrayViewType view() const
  {
    return m_view;
  }
  //! View on the option elements
  ArrayViewType view()
  {
    return m_view;
  }

  ConstReferenceType operator[](Integer i) const { return m_view[i]; }
  ReferenceType operator[](Integer i) { return m_view[i]; }

 public:

  ARCANE_CORE_EXPORT void print(const String& lang, std::ostream& o) const override;
  ICaseFunction* function() const override { return 0; }
  void updateFromFunction(Real, Integer) override {}

  ConstArrayView<T> values() const
  {
    const ArrayView<T>* v = this;
    return *v;
  }
  const T& value(Integer index) const { return this->operator[](index); }
  Integer size() const { return ArrayView<T>::size(); }
  ARCANE_CORE_EXPORT void visit(ICaseDocumentVisitor* visitor) const override;
  bool isPresent() const { return !m_view.empty(); }

 protected:

  void _search(bool is_phase1) override;
  virtual bool _allowPhysicalUnit();

 private:

  ArrayViewType m_view;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef CaseOptionSimpleT<Real> CaseOptionReal;
typedef CaseOptionSimpleT<Real2> CaseOptionReal2;
typedef CaseOptionSimpleT<Real3> CaseOptionReal3;
typedef CaseOptionSimpleT<Real2x2> CaseOptionReal2x2;
typedef CaseOptionSimpleT<Real3x3> CaseOptionReal3x3;
typedef CaseOptionSimpleT<bool> CaseOptionBool;
typedef CaseOptionSimpleT<Integer> CaseOptionInteger;
typedef CaseOptionSimpleT<Int32> CaseOptionInt32;
typedef CaseOptionSimpleT<Int64> CaseOptionInt64;
typedef CaseOptionSimpleT<String> CaseOptionString;

typedef CaseOptionSimpleT<RealArray> CaseOptionRealArray;
typedef CaseOptionSimpleT<Real2Array> CaseOptionReal2Array;
typedef CaseOptionSimpleT<Real3Array> CaseOptionReal3Array;
typedef CaseOptionSimpleT<Real2x2Array> CaseOptionReal2x2Array;
typedef CaseOptionSimpleT<Real3x3Array> CaseOptionReal3x3Array;
typedef CaseOptionSimpleT<BoolArray> CaseOptionBoolArray;
typedef CaseOptionSimpleT<IntegerArray> CaseOptionIntegerArray;
typedef CaseOptionSimpleT<Int32Array> CaseOptionInt32Array;
typedef CaseOptionSimpleT<Int64Array> CaseOptionInt64Array;
typedef CaseOptionSimpleT<StringArray> CaseOptionStringArray;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
