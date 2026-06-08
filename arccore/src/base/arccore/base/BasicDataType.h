// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicDataType.h                                             (C) 2000-2025 */
/*                                                                           */
/* Definition of basic data types.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BASICDATATYPE_H
#define ARCCORE_BASE_BASICDATATYPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Type of a basic data item.
 *
 * \note: The values must be contiguous and fit into one byte and must not be
 * modified because they are used for serialization, for example. However, it
 * is possible to add them.
 */
enum class eBasicDataType : unsigned char
{
  Unknown = 0, //!< Unknown or uninitialized data type
  Byte = 1, //!< Byte data type
  Float16 = 2, //!< Float16 data type
  Float32 = 3, //!< Float32 data type
  Float64 = 4, //!< Float64 data type
  Float128 = 5, //!< Float128 data type
  Int16 = 6, //!< 16-bit integer data type
  Int32 = 7, //!< 32-bit integer data type
  Int64 = 8, //!< 64-bit integer data type
  Int128 = 9, //!< 128-bit integer data type
  BFloat16 = 10, //! < BFloat16 data type
  Int8 = 11, //! 8-bit integer data type
  Real = Float64 //! Float64 data type
};
//! Number of supported basic types
constexpr unsigned char NB_BASIC_DATA_TYPE = 12;

//! Data type name.
extern "C++" ARCCORE_BASE_EXPORT const char*
basicDataTypeName(eBasicDataType type);

//! Finds the type associated with \a name
extern "C++" ARCCORE_BASE_EXPORT eBasicDataType
basicDataTypeFromName(const char* name, bool& has_error);

//! Finds the type associated with \a name. Throws an exception in case of error
extern "C++" ARCCORE_BASE_EXPORT eBasicDataType
basicDataTypeFromName(const char* name);

//! Size of data type \a type
extern "C++" ARCCORE_BASE_EXPORT Integer
basicDataTypeSize(eBasicDataType type);

//! Output operator for a float
extern "C++" ARCCORE_BASE_EXPORT std::ostream&
operator<<(std::ostream& ostr, eBasicDataType data_type);

//! Input operator from a float
extern "C++" ARCCORE_BASE_EXPORT std::istream&
operator>>(std::istream& istr, eBasicDataType& data_type);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Arcane::basicDataTypeFromName;
using Arcane::basicDataTypeName;
using Arcane::basicDataTypeSize;
using Arcane::eBasicDataType;
using Arcane::NB_BASIC_DATA_TYPE;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
