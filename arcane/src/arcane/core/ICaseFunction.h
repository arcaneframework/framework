// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseFunction.h                                             (C) 2000-2023 */
/*                                                                           */
/* Interface of a dataset function.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEFUNCTION_H
#define ARCANE_CORE_ICASEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 *
 * \brief Interface of a dataset function.
 *
 * \ingroup CaseOption
 *
 * A dataset function is a mathematical function f(x)->y where
 * \c x is the \e parameter and \c y is the \e value.
 *
 * In the current version, a function is piecewise described by
 * a set of (x,y) pairs.
 *
 * The methods that allow editing this lookup table are primarily used
 * by the dataset editor. In any case, they must not be called once the
 * complete dataset has been read (ICaseMng::readCaseOptions).
 */
class ARCANE_CORE_EXPORT ICaseFunction
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  /*!
   * \brief Type of a function parameter.
   */
  enum eParamType
  {
    ParamUnknown = 0, //!< Unknown parameter type
    ParamReal = 1, //!< Real type parameter
    ParamInteger = 2 //!< Integer type parameter
  };
  /*!
   * \brief Type of a function value
   */
  enum eValueType
  {
    ValueUnknown = 0, //!< Unknown value type
    ValueReal = 1, //!< Real type value
    ValueInteger = 2, //!< Integer type value
    ValueBool = 3, //!< Boolean type value
    ValueString = 4, //!< String type value
    ValueReal3 = 5 //!< 'Real3' type value
  };

 public:

  // NOTE: Temporarily leave this public destructor until
  // we call this destructor explicitly, but with the reference
  // counter, this should normally no longer be the case.
  virtual ~ICaseFunction() = default; //!< Releases resources

 public:

  /*! @name Function Name */
  //@{
  //! function name
  virtual String name() const = 0;

  //! Sets the function name to \a new_name
  virtual void setName(const String& new_name) = 0;
  //@}

  /*! @name Parameter Type */
  //@{
  //! Function parameter type
  virtual eParamType paramType() const = 0;

  //! Sets the function parameter type
  virtual void setParamType(eParamType type) = 0;
  //@}

  /*! @name Value Type */
  //@{
  //! Function value type
  virtual eValueType valueType() const = 0;

  //! Sets the function value type
  virtual void setValueType(eValueType type) = 0;
  //@}

  /*!
   * \brief Sets a value transformation function.
   * For now, this is just a multiplicative coefficient.
   * The string \a str must be convertible to the value type.
   */
  virtual void setTransformValueFunction(const String& str) = 0;

  //! Returns the value transformation function.
  virtual String transformValueFunction() const = 0;

  /*!
   * \brief Sets a parameter transformation function.
   * For now, this is just a multiplicative coefficient.
   * It is only applied to real parameters.
   * The string \a str must be convertible to a real number.
   */
  virtual void setTransformParamFunction(const String& str) = 0;

  //! Parameter transformation function
  virtual String transformParamFunction() const = 0;

  /*!
   * \brief Checks the validity of the function.
   * \retval true if the function is valid,
   * \retval false otherwise.
   */
  virtual bool checkIfValid() const = 0;

  /*!
   * \brief Sets the value of the deltat multiplier coefficient.
   *
   * This coefficient, 0.0 by default, is used for functions
   * that take physical time as a parameter. In this case,
   * the function uses the global current time as a parameter,
   * to which the global current time step multiplied
   * by this coefficient is added.
   */
  virtual void setDeltatCoef(Real v) = 0;

  //! Value of the deltat multiplier coefficient
  virtual Real deltatCoef() const = 0;

 public:

  //! Value \a v of the option for parameter \a param.
  virtual void value(Real param, Real& v) const = 0;

  //! Value \a v of the option for parameter \a param.
  virtual void value(Real param, Integer& v) const = 0;

  //! Value \a v of the option for parameter \a param.
  virtual void value(Real param, bool& v) const = 0;

  //! Value \a v of the option for parameter \a param.
  virtual void value(Real param, String& v) const = 0;

  //! Value \a v of the option for parameter \a param.
  virtual void value(Real param, Real3& v) const = 0;

  //! Value \a v of the option for parameter \a param.
  virtual void value(Integer param, Real& v) const = 0;

  //! Value \a v of the option for parameter \a param.
  virtual void value(Integer param, Integer& v) const = 0;

  // Value \a v of the option for parameter \a param.
  virtual void value(Integer param, bool& v) const = 0;

  //! Value \a v of the option for parameter \a param.
  virtual void value(Integer param, String& v) const = 0;

  //! Value \a v of the option for parameter \a param.
  virtual void value(Integer param, Real3& v) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
