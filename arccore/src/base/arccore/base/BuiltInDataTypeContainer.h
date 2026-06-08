// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BuiltInTypeContainer.h                                      (C) 2000-2024 */
/*                                                                           */
/* Container containing an instance of a class per data type.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BUILTINDATATYPECONTAINER_H
#define ARCCORE_BASE_BUILTINDATATYPECONTAINER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Container containing an instance of a class per data type.
 *
 * The template class \a Traits must contain a type \a InstanceType which
 * indicates the type of the container instance for each language type.
 * For example
 *
 * \code
 * template<typename DataType>
 * class Instancer
 * {
 *  public:
 *   using InstanceType = UniqueArray<DataType>;
 *   using value_type = DataType;
 * };
 * \endcode
 *
 * The instance() method allows retrieving a reference to the instance
 * from its type.
 */
template <template <typename DataType> class Traits>
class BuiltInDataTypeContainer
{
 public:

  using CharInstanceType = typename Traits<char>::InstanceType;
  using SignedCharInstanceType = typename Traits<signed char>::InstanceType;
  using UnsignedCharInstanceType = typename Traits<unsigned char>::InstanceType;
  using ShortInstanceType = typename Traits<short>::InstanceType;
  using UnsignedShortInstanceType = typename Traits<unsigned short>::InstanceType;
  using IntInstanceType = typename Traits<int>::InstanceType;
  using UnsignedIntInstanceType = typename Traits<unsigned int>::InstanceType;
  using LongInstanceType = typename Traits<long>::InstanceType;
  using UnsignedLongInstanceType = typename Traits<unsigned long>::InstanceType;
  using LongLongInstanceType = typename Traits<long long>::InstanceType;
  using UnsignedLongLongInstanceType = typename Traits<unsigned long long>::InstanceType;
  using FloatInstanceType = typename Traits<float>::InstanceType;
  using DoubleInstanceType = typename Traits<double>::InstanceType;
  using LongDoubleInstanceType = typename Traits<long double>::InstanceType;
  using BFloat16InstanceType = typename Traits<BFloat16>::InstanceType;
  using Float16InstanceType = typename Traits<Float16>::InstanceType;

 public:

  CharInstanceType& instance(char*) { return m_char; }
  SignedCharInstanceType& instance(signed char*) { return m_signed_char; }
  UnsignedCharInstanceType& instance(unsigned char*) { return m_unsigned_char; }
  ShortInstanceType& instance(short*) { return m_short; }
  UnsignedShortInstanceType& instance(unsigned short*) { return m_unsigned_short; }
  IntInstanceType& instance(int*) { return m_int; }
  UnsignedIntInstanceType& instance(unsigned int*) { return m_unsigned_int; }
  LongInstanceType& instance(long*) { return m_long; }
  UnsignedLongInstanceType& instance(unsigned long*) { return m_unsigned_long; }
  LongLongInstanceType& instance(long long*) { return m_long_long; }
  UnsignedLongLongInstanceType& instance(unsigned long long*) { return m_unsigned_long_long; }
  FloatInstanceType& instance(float*) { return m_float; }
  DoubleInstanceType& instance(double*) { return m_double; }
  LongDoubleInstanceType& instance(long double*) { return m_long_double; }
  BFloat16InstanceType& instance(BFloat16*) { return m_bfloat16; }
  Float16InstanceType& instance(Float16*) { return m_float16; }

 public:

  //! Applies the lambda function \a func to all containers
  template <typename Lambda>
  void apply(const Lambda& func)
  {
    func(m_char);
    func(m_signed_char);
    func(m_unsigned_char);
    func(m_short);
    func(m_unsigned_short);
    func(m_int);
    func(m_unsigned_int);
    func(m_long);
    func(m_unsigned_long);
    func(m_long_long);
    func(m_unsigned_long_long);
    func(m_float);
    func(m_double);
    func(m_long_double);
    func(m_bfloat16);
    func(m_float16);
  }

 private:

  CharInstanceType m_char = {};
  SignedCharInstanceType m_signed_char = {};
  UnsignedCharInstanceType m_unsigned_char = {};
  ShortInstanceType m_short = {};
  UnsignedShortInstanceType m_unsigned_short = {};
  IntInstanceType m_int = {};
  UnsignedIntInstanceType m_unsigned_int = {};
  LongInstanceType m_long = {};
  UnsignedLongInstanceType m_unsigned_long = {};
  LongLongInstanceType m_long_long = {};
  UnsignedLongLongInstanceType m_unsigned_long_long = {};
  FloatInstanceType m_float = {};
  DoubleInstanceType m_double = {};
  LongDoubleInstanceType m_long_double = {};
  BFloat16InstanceType m_bfloat16 = {};
  Float16InstanceType m_float16 = {};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
