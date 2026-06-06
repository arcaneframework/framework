// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypes.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Definition of data-related types.                                         */
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

// NOTE: Swig 4.2.0 does not handle 'Int32' well.
// (it works with Swig 4.1.1)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data type.
 */
#ifndef SWIG
enum eDataType : Int32
#else
enum eDataType
#endif
{
  DT_Byte = 0, //!< Byte data type
  DT_Real = 1, //!< Real data type
  DT_Int16 = 2, //!< 16-bit integer data type
  DT_Int32 = 3, //!< 32-bit integer data type
  DT_Int64 = 4, //!< 64-bit integer data type
  DT_String = 5, //!< UTF-8 character string data type
  DT_Real2 = 6, //!< Vector 2 data type
  DT_Real3 = 7, //!< Vector 3 data type
  DT_Real2x2 = 8, //!< 2x2 tensor data type
  DT_Real3x3 = 9, //!< 3x3 tensor data type
  DT_BFloat16 = 10, //!< 'BFloat16' data type
  DT_Float16 = 11, //!< 'Float16' data type
  DT_Float32 = 12, //!< 'Float32' data type
  DT_Int8 = 13, //!< 8-bit integer data type
  DT_Float128 = 14, //!< 128-bit floating point data type
  DT_Int128 = 15, //!< 128-bit integer data type
  DT_Unknown = 16 //!< Unknown or uninitialized data type
};

//! Number of eDataType values
static constexpr Int32 NB_ARCANE_DATA_TYPE = 17;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief List of names for eDataType.
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

//! Data type name.
extern "C++" ARCANE_CORE_EXPORT const char*
dataTypeName(eDataType type);

//! Finds the type associated with \a name
extern "C++" ARCANE_CORE_EXPORT eDataType
dataTypeFromName(const char* name, bool& has_error);

//! Finds the type associated with \a name. Throws an exception in case of error
extern "C++" ARCANE_CORE_EXPORT eDataType
dataTypeFromName(const char* name);

//! Size of data type \a type (which must be different from \a DT_String)
extern "C++" ARCANE_CORE_EXPORT Integer
dataTypeSize(eDataType type);

//! Output operator for a float
extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& ostr, eDataType data_type);

//! Input operator from a float
extern "C++" ARCANE_CORE_EXPORT std::istream&
operator>>(std::istream& istr, eDataType& data_type);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Possible data initialization policy.
 *
 * By default, for historical reasons, it is DIP_Legacy.
 *
 * The initialization policy is considered for
 * the initialization of Arcane variables. This value can be
 * set/retrieved via the functions
 * setGlobalDataInitialisationPolicy() and getGlobalDataInitialisationPolicy().
 *
 */
enum eDataInitialisationPolicy
{
  //! No forced initialization
  DIP_None = 0,

  /*!
   * \brief Initialization with the default constructor.
   *
   * For integers, this is the value 0. For reals, this is the value 0.0.
   */
  DIP_InitWithDefault = 1,

  /*!
   * \brief Initialization with NaN (Not a Number)
   *
   * This mode allows data of type Real and derived types (Real2, Real3, ...)
   * to be initialized with NaN values which trigger an exception when they
   * are used.
   */
  DIP_InitWithNan = 2,

  /*!
   * \brief Initialization in historical mode.
   *
   * This mode is kept for compatibility with versions of Arcane prior
   * to version 2.0 where it was the default mode.
   * In this mode, variables on mesh entities were always initialized with the default constructor
   * upon their first allocation, regardless of the value of
   * getGlobalDataInitialisationPolicy(). The initialization policy was only
   * considered upon a change in the number of elements (resize())
   * or for variables that were not mesh variables.
   */
  DIP_Legacy = 3,

  /*!
   * \brief Initialization with NaN upon creation and default constructor thereafter.
   *
   * This mode is identical to DIP_InitWithNan for variable creation
   * and to DIP_InitWithDefault when the variable size changes
   * (generally via a call to IVariable::resize() or IVariable::resizeFromGroup()).
   */
  DIP_InitInitialWithNanResizeWithDefault = 4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Sets the initialization policy for variables.
extern "C++" ARCANE_CORE_EXPORT void
setGlobalDataInitialisationPolicy(eDataInitialisationPolicy init_policy);

//! Gets the initialization policy for variables.
extern "C++" ARCANE_CORE_EXPORT eDataInitialisationPolicy
getGlobalDataInitialisationPolicy();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Possible trace type
enum eTraceType
{
  TT_None = 0,
  TT_Read = 1,
  TT_Write = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

// Macro to instantiate a template class for all numeric data types
#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE1(class_name) \
  template class class_name<Real>; \
  template class class_name<Real3>; \
  template class class_name<Real3x3>; \
  template class class_name<Real2>; \
  template class class_name<Real2x2>;

// Macro to instantiate a template class for all numeric data types
#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE2(class_name) \
  template class class_name<Int8>; \
  template class class_name<Int16>; \
  template class class_name<Int32>; \
  template class class_name<Int64>; \
  template class class_name<Byte>

// Macro to instantiate a template class for all numeric data types
#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE3(class_name) \
  template class class_name<BFloat16>; \
  template class class_name<Float16>; \
  template class class_name<Float32>;

// Macro to instantiate a template class for all numeric data types
#define ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(class_name) \
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE1(class_name); \
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE2(class_name); \
  ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE3(class_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
