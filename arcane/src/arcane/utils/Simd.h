// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Simd.h                                                      (C) 2000-2017 */
/*                                                                           */
/* Types for vectorization.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SIMD_H
#define ARCANE_UTILS_SIMD_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/SimdCommon.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file Simd.h
 *
 * This file contains the declarations of types used to manage
 * vectorization. Since several possible mechanisms exist, a choice must be made
 * based on the machine architecture and compilation options. Each mechanism uses
 * vectors of different sizes. In our case, since vectorization is
 * primarily used for calculations on doubles, the size of a vector will be
 * equal to the number of doubles that can fit into a vector.
 *
 * Currently, we support the following modes in order of priority. If
 * a mode is supported, the others are not used.
 * - AVX512 for Intel Knight Landing (KNL) or Xeon Skylake type architectures.
 *   The vector size in this mode is 8.
 * - AVX. For this mode, Arcane must be compiled with the option '--with-avx'.
 *   There are two modes: classic AVX and AVX2. For now, only the
 *   former is used, due to a lack of machines to test the latter. The
 *   vector size in this mode is 4.
 * - SSE. This mode is available by default because it exists on all
 *   x64 platforms. There are also several versions, and we limit ourselves
 *   to version 2. The vector size in this mode is 2.
 * - no mode. In this case, there is no specific vectorization.
 *   Nevertheless, to test the code, we allow emulation with
 *   vectors of size 2.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * The compilers (gcc and icc) define the following macros
 * on x64 depending on vectorization support:
 * : avx: __AVX__
 * : avx2: __AVX2__
 * : avx512f: __AVX512F__
 * : sse2: __SSE2__
 *
 * Note that for avx2 with gcc, FMA is not enabled by default.
 * For example:
 * - gcc -mavx2 : no fma
 * - gcc -mavx2 -mfma : fma active
 * - gcc -march=haswell : fma active
 */

// Simd via emulation.
#include "arcane/utils/SimdEMUL.h"

// Add SSE support if available
#if (defined(_M_X64) || defined(__x86_64__)) && !defined(ARCANE_NO_SSE)
// SSE2 is available on all x64 CPUs
// The macro __x64_64__ is defined for Linux machines
// The macro _M_X64 is defined for Windows machines
#define ARCANE_HAS_SSE
#include <emmintrin.h>
#include "arcane/utils/SimdSSE.h"
#endif

// Add AVX support if available
#if defined(ARCANE_HAS_AVX) || defined(ARCANE_HAS_AVX512)
#include <immintrin.h>
#include "arcane/utils/SimdAVX.h"
#endif

// Add AVX512 support if available
#if defined(ARCANE_HAS_AVX512)
#include <immintrin.h>
#include "arcane/utils/SimdAVX512.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Macro to iterate over the indices of a SIMD real or derived vector (Real2, Real3, ...).
 */
#define ENUMERATE_SIMD_REAL(_iter) \
  for (::Arcane::Integer _iter(0); _iter < SimdReal ::BLOCK_SIZE; ++_iter)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Defines the SimdInfo type based on available vectorization.
 * It takes the type that allows the most vectorization.
 */
#if defined(ARCANE_HAS_AVX512)
typedef AVX512SimdInfo SimdInfo;
#elif defined(ARCANE_HAS_AVX)
typedef AVXSimdInfo SimdInfo;
#elif defined(ARCANE_HAS_SSE)
typedef SSESimdInfo SimdInfo;
#else
typedef EMULSimdInfo SimdInfo;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief SIMD vector of real numbers.
 */
typedef SimdInfo::SimdReal SimdReal;
const int SimdSize = SimdReal::Length;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Represents a vectorized Real3.
 */
class ARCANE_UTILS_EXPORT SimdReal3
{
 public:

  typedef SimdReal::Int32IndexType Int32IndexType;

 public:

  SimdReal x;
  SimdReal y;
  SimdReal z;
  SimdReal3() {}
  SimdReal3(SimdReal _x, SimdReal _y, SimdReal _z)
  : x(_x)
  , y(_y)
  , z(_z)
  {}
  SimdReal3(const Real3* base, const Int32IndexType& idx)
  {
    for (Integer i = 0, n = SimdReal::BLOCK_SIZE; i < n; ++i) {
      Real3 v = base[idx[i]];
      this->set(i, v);
    }
  }
  const Real3 operator[](Integer i) const { return Real3(x[i], y[i], z[i]); }

  void set(Real3* base, const Int32IndexType& idx) const
  {
    for (Integer i = 0, n = SimdReal::BLOCK_SIZE; i < n; ++i) {
      base[idx[i]] = this->get(i);
    }
  }

  // TODO: rename this method
  void set(Integer i, Real3 r)
  {
    x[i] = r.x;
    y[i] = r.y;
    z[i] = r.z;
  }
  Real3 get(Integer i) const
  {
    return Real3(x[i], y[i], z[i]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Represents a vectorized Real2.
 */
class ARCANE_UTILS_EXPORT SimdReal2
{
 public:

  typedef SimdReal::Int32IndexType Int32IndexType;

 public:

  SimdReal x;
  SimdReal y;
  SimdReal2() {}
  SimdReal2(SimdReal _x, SimdReal _y)
  : x(_x)
  , y(_y)
  {}
  SimdReal2(const Real2* base, const Int32IndexType& idx)
  {
    for (Integer i = 0, n = SimdReal::BLOCK_SIZE; i < n; ++i) {
      Real2 v = base[idx[i]];
      this->set(i, v);
    }
  }
  const Real2 operator[](Integer i) const { return Real2(x[i], y[i]); }

  void set(Real2* base, const Int32IndexType& idx) const
  {
    for (Integer i = 0, n = SimdReal::BLOCK_SIZE; i < n; ++i) {
      base[idx[i]] = this->get(i);
    }
  }

  void set(Integer i, Real2 r)
  {
    x[i] = r.x;
    y[i] = r.y;
  }
  Real2 get(Integer i) const
  {
    return Real2(x[i], y[i]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Represents a vectorized Real3x3.
 */
class ARCANE_UTILS_EXPORT SimdReal3x3
{
 public:

  typedef SimdReal::Int32IndexType Int32IndexType;

 public:

  SimdReal3 x;
  SimdReal3 y;
  SimdReal3 z;
  SimdReal3x3() {}
  SimdReal3x3(SimdReal3 _x, SimdReal3 _y, SimdReal3 _z)
  : x(_x)
  , y(_y)
  , z(_z)
  {}
  SimdReal3x3(const Real3x3* base, const Int32IndexType& idx)
  {
    for (Integer i = 0, n = SimdReal::BLOCK_SIZE; i < n; ++i) {
      Real3x3 v = base[idx[i]];
      this->set(i, v);
    }
  }
  const Real3x3 operator[](Integer i) const { return Real3x3(x[i], y[i], z[i]); }

  void set(Real3x3* base, const Int32IndexType& idx) const
  {
    for (Integer i = 0, n = SimdReal::BLOCK_SIZE; i < n; ++i) {
      base[idx[i]] = this->get(i);
    }
  }

  // TODO: rename this method
  void set(Integer i, Real3x3 r)
  {
    x.set(i, r.x);
    y.set(i, r.y);
    z.set(i, r.z);
  }
  Real3x3 get(Integer i) const
  {
    return Real3x3(x[i], y[i], z[i]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Represents a vectorized Real2x2.
 */
class ARCANE_UTILS_EXPORT SimdReal2x2
{
 public:

  typedef SimdReal::Int32IndexType Int32IndexType;

 public:

  SimdReal2 x;
  SimdReal2 y;
  SimdReal2x2() {}
  SimdReal2x2(SimdReal2 _x, SimdReal2 _y)
  : x(_x)
  , y(_y)
  {}
  SimdReal2x2(const Real2x2* base, const Int32IndexType& idx)
  {
    for (Integer i = 0, n = SimdReal::BLOCK_SIZE; i < n; ++i) {
      Real2x2 v = base[idx[i]];
      this->set(i, v);
    }
  }
  const Real2x2 operator[](Integer i) const { return Real2x2(x[i], y[i]); }

  void set(Real2x2* base, const Int32IndexType& idx) const
  {
    for (Integer i = 0, n = SimdReal::BLOCK_SIZE; i < n; ++i) {
      base[idx[i]] = this->get(i);
    }
  }

  // TODO: rename this method
  void set(Integer i, Real2x2 r)
  {
    x.set(i, r.x);
    y.set(i, r.y);
  }
  Real2x2 get(Integer i) const
  {
    return Real2x2(x[i], y[i]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Characteristics of vector types.
 *
 * Default instantiation for types that do not have a corresponding vector type.
 * Currently, only the types 'Real', 'Real2', and 'Real3' have one.
 */
template <typename DataType>
class SimdTypeTraits
{
 public:

  typedef void SimdType;
};

template <>
class SimdTypeTraits<Real>
{
 public:

  typedef SimdReal SimdType;
};

template <>
class SimdTypeTraits<Real2>
{
 public:

  typedef SimdReal2 SimdType;
};

template <>
class SimdTypeTraits<Real2x2>
{
 public:

  typedef SimdReal2x2 SimdType;
};

template <>
class SimdTypeTraits<Real3>
{
 public:

  typedef SimdReal3 SimdType;
};

template <>
class SimdTypeTraits<Real3x3>
{
 public:

  typedef SimdReal3x3 SimdType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Base class for vector enumerators with indirection.
 *
 * \warning The arrays of local indices (\a local_ids) passed to the
 * constructors must be aligned.
 */
class ARCANE_UTILS_EXPORT SimdEnumeratorBase
{
 public:

  typedef SimdInfo::SimdInt32IndexType SimdIndexType;

 public:

  SimdEnumeratorBase()
  : m_local_ids(nullptr)
  , m_index(0)
  , m_count(0)
  {}
  SimdEnumeratorBase(const Int32* local_ids, Integer n)
  : m_local_ids(local_ids)
  , m_index(0)
  , m_count(n)
  {
    _checkValid();
  }
  explicit SimdEnumeratorBase(Int32ConstArrayView local_ids)
  : m_local_ids(local_ids.data())
  , m_index(0)
  , m_count(local_ids.size())
  {
    _checkValid();
  }

 public:

  bool hasNext() { return m_index < m_count; }

  //! Local indices
  const Int32* unguardedLocalIds() const { return m_local_ids; }

  void operator++() { m_index += SimdSize; }

  /*!
   * \brief Number of valid values for the current iterator.
   * \pre hasNext()==true
   */
  inline Integer nbValid() const
  {
    Integer nb_valid = (m_count - m_index);
    if (nb_valid > SimdSize)
      nb_valid = SimdSize;
    return nb_valid;
  }

  Integer count() const { return m_count; }

 protected:

  const Int32* ARCANE_RESTRICT m_local_ids;
  Integer m_index;
  Integer m_count;

  const SimdIndexType* ARCANE_RESTRICT
  _currentSimdIndex() const
  {
    return (const SimdIndexType*)(m_local_ids + m_index);
  }

 private:

  // Checks that m_local_ids is correctly aligned.
  void _checkValid()
  {
#ifdef ARCANE_SIMD_BENCH
    Int64 modulo = (Int64)(m_local_ids) % SimdIndexType::Alignment;
    if (modulo != 0) {
      throw BadAlignmentException();
    }
#else
    _checkValidHelper();
#endif
  }
  void _checkValidHelper();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
