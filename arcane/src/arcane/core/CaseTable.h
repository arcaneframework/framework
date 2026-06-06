// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseTable.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Class managing a lookup table.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASETABLE_H
#define ARCANE_CORE_CASETABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/datatype/SmallVariant.h"

#include "arcane/core/CaseFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseTableParams;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Data set function.
 */
class ARCANE_CORE_EXPORT CaseTable
: public CaseFunction
{
 public:

  /*!
   * \brief Types of errors returned by the class.
   */
  enum eError
  {
    ErrNo,
    //! Indicates that an element index is not valid
    ErrBadRange,
    //! Indicates that converting the parameter to the desired type is impossible
    ErrCanNotConvertParamToRightType,
    //! Indicates that converting the value to the desired type is impossible
    ErrCanNotConvertValueToRightType,
    //! Indicates that the parameter is not greater than the previous one
    ErrNotGreaterThanPrevious,
    //! Indicates that the parameter is not less than the next one
    ErrNotLesserThanNext
  };

  /*! \brief Type of the table curve.
   */
  enum eCurveType
  {
    CurveUnknown = 0, //!< Unknown curve type
    CurveConstant = 1, //!< Piecewise constant curve
    CurveLinear = 2 //!< Piecewise linear curve
  };

 public:

  /*! \brief Constructs a lookup table from the data set.
   * \param curve_type type of the lookup table curve
   */
  CaseTable(const CaseFunctionBuildInfo& info, eCurveType curve_type);
  virtual ~CaseTable();

 public:

  //! Number of elements in the function
  virtual Integer nbElement() const;

  //! The i-th value in the string \a str
  virtual void valueToString(Integer id, String& str) const;

  //! The i-th parameter in the string \a str
  virtual void paramToString(Integer id, String& param) const;

  /*!
   * \brief Modifies the parameter of element \a id.
   *
   * Uses \a value as the new value for the parameter.
   * \a value must be convertible to the parameter type.
   *
   * \return the error value, ErrNo otherwise.
   */
  virtual eError setParam(Integer id, const String& value);

  /*!
   * \brief Modifies the value of element \a id.
   *
   * Uses \a value as the new value.
   * \a value must be convertible to the value type.
   *
   * \return the error value, ErrNo otherwise.
   */
  virtual eError setValue(Integer id, const String& value);

  /*! \brief Adds an element to the table.
   *
   * Adds the element (param,value) to the table.
   *
   * \return the error value, ErrNo otherwise.
   */
  virtual eError appendElement(const String& param, const String& value);

  /*!
   * \brief Inserts a couple (parameter,value) into the function.
   *
   * Inserts a couple (parameter,value) identical to the one found at position
   * \a id. Subsequent parameters are shifted by one position. It is then possible
   * to modify this couple using the methods setParam() or setValue().
   *
   * If \a id is greater than the number of elements in the function, an element
   * is added to the end with the same value as the last element of
   * the function.
   */
  virtual void insertElement(Integer id);

  /*!
   * \brief Removes a couple (parameter,value) from the function.
   *
   * If \a id is greater than the number of elements in the function, no
   * operation is performed.
   */
  virtual void removeElement(Integer id);

  /*! @name Curve Type */
  //@{
  //! Returns the curve type of the function
  virtual eCurveType curveType() const { return m_curve_type; }
  //@}

  virtual void setParamType(eParamType type);

  virtual bool checkIfValid() const;

  virtual void value(Real param, Real& v) const;
  virtual void value(Real param, Integer& v) const;
  virtual void value(Real param, bool& v) const;
  virtual void value(Real param, String& v) const;
  virtual void value(Real param, Real3& v) const;
  virtual void value(Integer param, Real& v) const;
  virtual void value(Integer param, Integer& v) const;
  virtual void value(Integer param, bool& v) const;
  virtual void value(Integer param, String& v) const;
  virtual void value(Integer param, Real3& v) const;

 public:
 private:

  CaseTableParams* m_param_list;
  UniqueArray<SmallVariant> m_value_list; //!< List of values.
  eCurveType m_curve_type; //!< Curve type
  bool m_use_fast_search = true;

 private:

  template <typename U, typename V> void _findValue(U param, V& value) const;
  template <typename U, typename V> void _findValueAndApplyTransform(U param, V& value) const;

  bool _isValidIndex(Integer index) const;
  eError _setValue(Integer index, const String& value_str);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
