// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueFiller.h                                               (C) 2000-2024 */
/*                                                                           */
/* Fonctions pour générer des valeurs (pour les tests).                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_VALUEFILLER_H
#define ARCCORE_BASE_VALUEFILLER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"
#include "arccore/base/BFloat16.h"
#include "arccore/base/Float16.h"
#include "arccore/base/Float128.h"
#include "arccore/base/Int128.h"

#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::ValueFiller::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class BuildInfo
{
 public:

  using ClampValue = std::pair<DataType, DataType>;
  BuildInfo(const ClampValue& clamp_value)
  : m_clamp_value(clamp_value)
  {}
  ClampValue clampValue() const
  {
    return m_clamp_value;
  }

 private:

  ClampValue m_clamp_value;
};

template <typename DataType>
BuildInfo<DataType> _getFloatBuildInfo();
template <> BuildInfo<Float128> _getFloatBuildInfo<Float128>()
{
  return { { -122334353.245, +983973536.324 } };
}
template <> BuildInfo<long double> _getFloatBuildInfo<long double>()
{
  return { { -334353.245, +73536.324 } };
}
template <> BuildInfo<double> _getFloatBuildInfo<double>()
{
  return { { -334353.245, +73536.324 } };
}
template <> BuildInfo<float> _getFloatBuildInfo<float>()
{
  return { { -14353.245f, 3536.324f } };
}
template <> BuildInfo<BFloat16> _getFloatBuildInfo<BFloat16>()
{
  return { { BFloat16{ -35321.2f }, BFloat16{ 63236.3f } } };
}
template <> BuildInfo<Float16> _getFloatBuildInfo<Float16>()
{
  return { { Float16{ -353.2f }, Float16{ 636.3f } } };
}
template <typename DataType> BuildInfo<DataType> _getIntegerBuildInfo(std::true_type)
{
  if constexpr (sizeof(DataType) == 1)
    return { { -25, 32 } };
  if constexpr (sizeof(DataType) == 2)
    return { { -212, 345 } };
  if constexpr (sizeof(DataType) == 4)
    return { { -14353, 12345 } };
  if constexpr (sizeof(DataType) == 8)
    return { { -234353, 452345 } };
  if constexpr (sizeof(DataType) == 16)
    return { { -33435332, 9232023 } };
}
template <typename DataType> BuildInfo<DataType> _getIntegerBuildInfo(std::false_type)
{
  if constexpr (sizeof(DataType) == 1)
    return { { 0, 89 } };
  if constexpr (sizeof(DataType) == 2)
    return { { 0, 721 } };
  if constexpr (sizeof(DataType) == 4)
    return { { 0, 29540 } };
  if constexpr (sizeof(DataType) == 8)
    return { { 0, 1290325 } };
  if constexpr (sizeof(DataType) == 16)
    return { { 0, 931290325 } };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class FillTraitsT;

template <typename DataType>
class FillTraitsBaseT
{
 public:

  using value_type = DataType;

 public:

  explicit FillTraitsBaseT(const impl::BuildInfo<DataType>& build_info)
  : m_min_clamp(build_info.clampValue().first)
  , m_max_clamp(build_info.clampValue().second)
  {}

 public:

  DataType minClamp() const { return m_min_clamp; }
  DataType maxClamp() const { return m_max_clamp; }

 private:

  DataType m_min_clamp = {};
  DataType m_max_clamp = {};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename RandomGeneratorDataType = DataType>
class IntegerFillTraitsT
: public FillTraitsBaseT<DataType>
{
  using BaseClass = FillTraitsBaseT<DataType>;

 public:

  using UniformGeneratorType = std::uniform_int_distribution<RandomGeneratorDataType>;

 public:

  IntegerFillTraitsT()
  : BaseClass(impl::_getIntegerBuildInfo<DataType>(std::is_signed<DataType>{}))
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename RandomGeneratorDataType = DataType>
class FloatFillTraitsT
: public FillTraitsBaseT<DataType>
{
  using BaseClass = FillTraitsBaseT<DataType>;

 public:

  using UniformGeneratorType = std::uniform_real_distribution<RandomGeneratorDataType>;

 public:

  explicit FloatFillTraitsT()
  : BaseClass(impl::_getFloatBuildInfo<DataType>())
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class FillTraitsT<char>
: public IntegerFillTraitsT<char,std::conditional<std::is_signed_v<char>,short,unsigned short>::type>
{
};

template <>
class FillTraitsT<signed char>
: public IntegerFillTraitsT<signed char,short>
{
};

template <>
class FillTraitsT<unsigned char>
: public IntegerFillTraitsT<unsigned char,unsigned short>
{
};

template <>
class FillTraitsT<short>
: public IntegerFillTraitsT<short>
{
};

template <>
class FillTraitsT<unsigned short>
: public IntegerFillTraitsT<unsigned short>
{
};

template <>
class FillTraitsT<int>
: public IntegerFillTraitsT<int>
{
};

template <>
class FillTraitsT<unsigned int>
: public IntegerFillTraitsT<unsigned int>
{
};

template <>
class FillTraitsT<long>
: public IntegerFillTraitsT<long>
{
};

template <>
class FillTraitsT<unsigned long>
: public IntegerFillTraitsT<unsigned long>
{
};

template <>
class FillTraitsT<long long>
: public IntegerFillTraitsT<long long>
{
};

template <>
class FillTraitsT<unsigned long long>
: public IntegerFillTraitsT<unsigned long long>
{
};


template <>
class FillTraitsT<Int128>
: public IntegerFillTraitsT<Int128, Int64>
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class FillTraitsT<float>
: public FloatFillTraitsT<float>
{
};

template <>
class FillTraitsT<double>
: public FloatFillTraitsT<double>
{
};

template <>
class FillTraitsT<long double>
: public FloatFillTraitsT<long double>
{
};

template <>
class FillTraitsT<BFloat16>
: public FloatFillTraitsT<BFloat16, float>
{
};

template <>
class FillTraitsT<Float16>
: public FloatFillTraitsT<Float16, float>
{
};

template <>
class FillTraitsT<Float128>
: public FloatFillTraitsT<Float128, long double>
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::ValueFiller::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::ValueFiller
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> inline
void fillRandom(Int64 seed, Span<DataType> values)
{
  std::seed_seq rng_seed{ seed };
  using TraitsType = impl::FillTraitsT<DataType>;
  using UniformGeneratorType = typename TraitsType::UniformGeneratorType;
  TraitsType traits_value;
  std::mt19937_64 randomizer(rng_seed);
  UniformGeneratorType rng_distrib(traits_value.minClamp(), traits_value.maxClamp());

  Int64 n1 = values.size();
  for (Int32 i = 0; i < n1; ++i) {
    values[i] = static_cast<DataType>(rng_distrib(randomizer));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::ValueFiller

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

