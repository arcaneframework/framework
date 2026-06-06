// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypeTraits.h                                            (C) 2000-2025 */
/*                                                                           */
/* Data type characteristics.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_DATATYPETRAITS_H
#define ARCANE_DATATYPE_DATATYPETRAITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/BasicDataType.h"
#include "arcane/utils/BFloat16.h"
#include "arcane/utils/Float16.h"
#include "arcane/utils/Float128.h"
#include "arcane/utils/Int128.h"

#include "arcane/core/ArcaneTypes.h"

/*
 * NOTE: Functions such as HasSubscriptOperator(), HasComponent*()
 * have not been used since December 2025. Previously, they were used
 * by DataViewSetter and DataViewGetterSetter, but this is no longer the case with the transition to C++20.
 * The same applies to the types ComponentType, SubscriptType, FunctionCall1ReturnType, or FunctionCall2ReturnType.
 * We can therefore eventually deprecate these types and then remove them.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DataTypeScalarReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class DataTypeTraitsT;
class Real2Proxy;
class Real3Proxy;
class Real3x3Proxy;
class Real2x2Proxy;
template<typename Type>
class BuiltInProxy;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type \c bool.
 */
template<>
class DataTypeTraitsT<bool>
{
 public:

  //! Data type
  typedef bool Type;

  //! Base data type of this data type
  typedef bool BasicType;

  //! Number of elements of the base type
  static constexpr int nbBasicType() { return 1; }

  //! Data type name
  static constexpr const char* name() { return "Bool"; }

  /*! Data type
   * \todo: create DT_Bool type instead.
   */
  static constexpr eDataType type() { return DT_Byte; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Byte; }

  //! Associated proxy type
  typedef BuiltInProxy<bool> ProxyType;

  //! Element initialized to NAN
  static Type nanValue() { return 0; }

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return false; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type \c Byte.
 */
template<>
class DataTypeTraitsT<Byte>
{
 public:

  //! Data type
  typedef Byte Type;

  //! Base data type of this data type
  typedef Byte BasicType;

  //! Number of elements of the base type
  static constexpr int nbBasicType() { return 1; }

  //! Data type name
  static constexpr const char* name() { return "Byte"; }

  //! Data type
  static constexpr eDataType type() { return DT_Byte; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Byte; }

  //! Associated proxy type
  typedef BuiltInProxy<Byte> ProxyType;

  //! Element initialized to NAN
  static constexpr Type nanValue() { return 0; }

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type \c Real.
 */
template<>
class DataTypeTraitsT<Real>
{
 public:

  //! Data type
  typedef Real Type;

  //! Base data type of this data type
  typedef Real BasicType;

  //! Number of elements of the base type
  static constexpr int nbBasicType() { return 1; }

  //! Data type name
  static constexpr const char* name() { return "Real"; }

  //! Data type
  static constexpr eDataType type() { return DT_Real; }

  //! Base data type.
  // TODO: automatically calculate the size
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Float64; }

  //! Associated proxy type
  typedef BuiltInProxy<Real> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return 0.0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type \c Float128
 */
template<>
class DataTypeTraitsT<Float128>
{
 public:

  //! Data type
  typedef Float128 Type;

  //! Base data type of this data type
  typedef Float128 BasicType;

  //! Number of elements of the base type
  static constexpr int nbBasicType() { return 1; }

  //! Data type name
  static constexpr const char* name() { return "Float128"; }

  //! Data type
  static constexpr eDataType type() { return DT_Float128; }

  //! Base data type.
  // TODO: automatically calculate the size
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Float128; }

  //! Associated proxy type
  typedef BuiltInProxy<Float128> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return Float128(0.0l); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type \c Float32.
 */
template<>
class DataTypeTraitsT<Float32>
{
 public:

  //! Data type
  typedef Float32 Type;

  //! Base data type of this data type
  typedef Float32 BasicType;

  //! Number of elements of the base type
  static constexpr int nbBasicType() { return 1; }

  //! Data type name
  static constexpr const char* name() { return "Float32"; }

  //! Data type
  static constexpr eDataType type() { return DT_Float32; }

  //! Base data type.
  // TODO: automatically calculate the size
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Float32; }

  //! Associated proxy type
  typedef BuiltInProxy<Float32> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return 0.0f; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type \c BFloat16.
 */
template<>
class DataTypeTraitsT<BFloat16>
{
 public:

  //! Data type
  typedef BFloat16 Type;

  //! Base data type of this data type
  typedef BFloat16 BasicType;

  //! Number of elements of the base type
  static constexpr int nbBasicType() { return 1; }

  //! Data type name
  static constexpr const char* name() { return "BFloat16"; }

  //! Data type
  static constexpr eDataType type() { return DT_BFloat16; }

  //! Base data type.
  // TODO: automatically calculate the size
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::BFloat16; }

  //! Associated proxy type
  typedef BuiltInProxy<BFloat16> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return {}; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type \c Float16.
 */
template<>
class DataTypeTraitsT<Float16>
{
 public:

  //! Data type
  typedef Float16 Type;

  //! Base data type of this data type
  typedef Float16 BasicType;

  //! Number of elements of the base type
  static constexpr int nbBasicType() { return 1; }

  //! Data type name
  static constexpr const char* name() { return "Float16"; }

  //! Data type
  static constexpr eDataType type() { return DT_Float16; }

  //! Base data type.
  // TODO: automatically calculate the size
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Float16; }

  //! Associated proxy type
  typedef BuiltInProxy<Float16> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return {}; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>Integer</tt>.
 */
template<>
class DataTypeTraitsT<Int8>
{
 public:

  //! Data type
  typedef Int8 Type;

  //! Base data type of this data type
  typedef Int8 BasicType;

  //! Number of elements of the base type
  static constexpr int nbBasicType() { return 1; }

  //! Data type name
  static constexpr const char* name() { return "Int8"; }

  //! Data type
  static constexpr eDataType type() { return DT_Int8; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Int8; }

  //! Associated proxy type
  typedef BuiltInProxy<Int8> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>Integer</tt>.
 */
template<>
class DataTypeTraitsT<Int16>
{
 public:

  //! Data type
  typedef Int16 Type;

  //! Base data type of this data type
  typedef Int16 BasicType;

  //! Number of elements of the base type
  static constexpr int nbBasicType() { return 1; }

  //! Data type name
  static constexpr const char* name() { return "Int16"; }

  //! Data type
  static constexpr eDataType type() { return DT_Int16; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Int16; }

  //! Associated proxy type
  typedef BuiltInProxy<Int32> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>Int32</tt>.
 */
template<>
class DataTypeTraitsT<Int32>
{
 public:

  //! Data type
  typedef Int32 Type;

  //! Base data type of this data type
  typedef Int32 BasicType;

  //! Number of base type elements
  static constexpr int nbBasicType() { return 1; }

  //! Name of the data type
  static constexpr const char* name() { return "Int32"; }

  //! Data type
  static constexpr eDataType type() { return DT_Int32; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Int32; }

  //! Type of associated proxy
  typedef BuiltInProxy<Int32> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>Int64</tt>.
 */
template<>
class DataTypeTraitsT<Int64>
{
 public:

  //! Data type
  typedef Int64 Type;

  //! Base data type of this data type
  typedef Int64 BasicType;

  //! Number of base type elements
  static constexpr int nbBasicType() { return 1; }

  //! Name of the data type
  static constexpr const char* name() { return "Int64"; }

  //! Data type
  static constexpr eDataType type() { return DT_Int64; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Int64; }

  //! Type of associated proxy
  typedef BuiltInProxy<Int64> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>Int128</tt>.
 */
template<>
class DataTypeTraitsT<Int128>
{
 public:

  //! Data type
  typedef Int128 Type;

  //! Base data type of this data type
  typedef Int128 BasicType;

  //! Number of base type elements
  static constexpr int nbBasicType() { return 1; }

  //! Name of the data type
  static constexpr const char* name() { return "Int128"; }

  //! Data type
  static constexpr eDataType type() { return DT_Int128; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return eBasicDataType::Int128; }

  //! Type of associated proxy
  typedef BuiltInProxy<Int128> ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static constexpr Type defaultValue() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>String</tt>.
 */
template<>
class DataTypeTraitsT<String>
{
 public:

  //! Data type
  typedef String Type;

  //! Base data type of this data type
  typedef String BasicType;

  //! Number of base type elements
  static constexpr int nbBasicType() { return 1; }

  //! Name of the data type
  static constexpr const char* name() { return "String"; }

  //! Data type
  static constexpr eDataType type() { return DT_String; }

  //! Type of associated proxy
  typedef String ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static ARCANE_CORE_EXPORT Type defaultValue();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>Real2</tt>.
 */
template<>
class DataTypeTraitsT<Real2>
{
 public:

  //! Data type
  typedef Real2 Type;

  //! Base data type of this data type
  typedef Real BasicType;

  //! Return type of operator[] for this type
  using SubscriptType = Real;

  //! Number of base type elements
  static constexpr int nbBasicType() { return 2; }

  //! Name of the data type
  static constexpr const char* name() { return "Real2"; }

  //! Data type
  static constexpr eDataType type() { return DT_Real2; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }

  //! Type of associated proxy
  typedef Real2Proxy ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static ARCANE_CORE_EXPORT Type defaultValue();

 public:

  static constexpr bool HasSubscriptOperator() { return true; }
  static constexpr bool HasComponentX() { return true; }
  static constexpr bool HasComponentY() { return true; }

  using ComponentType = Real;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>Real3</tt>.
 */
template<>
class DataTypeTraitsT<Real3>
{
 public:

  //! Data type
  typedef Real3 Type;

  //! Base data type of this data type
  typedef Real BasicType;

  //! Return type of operator[] for this type
  using SubscriptType = Real;

  //! Number of base type elements
  static constexpr int nbBasicType() { return 3; }

  //! Name of the data type
  static constexpr const char* name() { return "Real3"; }

  //! Data type
  static constexpr eDataType type() { return DT_Real3; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }

  //! Type of associated proxy
  typedef Real3Proxy ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static ARCANE_CORE_EXPORT Type defaultValue();

  static constexpr bool HasSubscriptOperator() { return true; }
  static constexpr bool HasComponentX() { return true; }
  static constexpr bool HasComponentY() { return true; }
  static constexpr bool HasComponentZ() { return true; }

  using ComponentType = Real;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>Real2x2</tt>.
 */
template<>
class DataTypeTraitsT<Real2x2>
{
 public:

  //! Data type
  typedef Real2x2 Type;

  //! Base data type of this data type
  typedef Real BasicType;

  //! Return type of operator[] for this type
  using SubscriptType = Real2;

  //! Number of base type elements
  static constexpr int nbBasicType() { return 4; }

  //! Name of the data type
  static constexpr const char* name() { return "Real2x2"; }

  //! Data type
  static constexpr eDataType type() { return DT_Real2x2; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }

  //! Type of associated proxy
  typedef Real2x2Proxy ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static ARCANE_CORE_EXPORT Type defaultValue();

  static constexpr bool HasSubscriptOperator() { return true; }

  static constexpr bool HasComponentXX() { return true; }
  static constexpr bool HasComponentYX() { return true; }
  static constexpr bool HasComponentXY() { return true; }
  static constexpr bool HasComponentYY() { return true; }

  static constexpr bool HasComponentX() { return true; }
  static constexpr bool HasComponentY() { return true; }

  using ComponentType = Real2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>Real3x3</tt>.
 */
template<>
class DataTypeTraitsT<Real3x3>
{
 public:

  //! Data type
  typedef Real3x3 Type;

  //! Base data type of this data type
  typedef Real BasicType;

  //! Return type of operator[] for this type
  using SubscriptType = Real3;

  //! Number of base type elements
  static constexpr int nbBasicType() { return 9; }

  //! Name of the data type
  static constexpr const char* name() { return "Real3x3"; }

  //! Data type
  static constexpr eDataType type() { return DT_Real3x3; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }

  //! Type of associated proxy
  typedef Real3x3Proxy ProxyType;

  //! Fills the elements of \a values with Nan.
  static void fillNan(ArrayView<Type> values);

  //! Default value.
  static ARCANE_CORE_EXPORT Type defaultValue();

  static constexpr bool HasSubscriptOperator() { return true; }

  static constexpr bool HasComponentXX() { return true; }
  static constexpr bool HasComponentYX() { return true; }
  static constexpr bool HasComponentZX() { return true; }
  static constexpr bool HasComponentXY() { return true; }
  static constexpr bool HasComponentYY() { return true; }
  static constexpr bool HasComponentZY() { return true; }
  static constexpr bool HasComponentXZ() { return true; }
  static constexpr bool HasComponentYZ() { return true; }
  static constexpr bool HasComponentZZ() { return true; }

  static constexpr bool HasComponentX() { return true; }
  static constexpr bool HasComponentY() { return true; }
  static constexpr bool HasComponentZ() { return true; }

  using ComponentType = Real3;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>NumVector<Real,Size></tt>.
 */
template<int Size>
class DataTypeTraitsT<NumVector<Real,Size>>
{
 public:

  //! Data type
  typedef Real Type;

  //! Base data type of this data type
  typedef Real BasicType;

  //! Return type of operator()(Int32) for this type
  using FunctionCall1ReturnType = Real;

  //! Number of base type elements
  static constexpr int nbBasicType() { return Size; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Specialization of DataTypeTraitsT for the type <tt>NumMatrix<Real,RowSize,ColumnSize></tt>.
 */
template<int RowSize,int ColumnSize>
class DataTypeTraitsT<NumMatrix<Real,RowSize,ColumnSize>>
{
 public:

  //! Data type
  typedef Real Type;

  //! Base data type of this data type
  typedef Real BasicType;

  //! Return type of operator()(Int32,Int32) for this type
  using FunctionCall2ReturnType = Real;

  //! Number of base type elements
  static constexpr int nbBasicType() { return RowSize * ColumnSize; }

  //! Base data type.
  static constexpr eBasicDataType basicDataType() { return DataTypeTraitsT<Real>::basicDataType(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
