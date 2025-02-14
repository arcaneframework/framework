// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FloatConversion.h                                           (C) 2000-2025 */
/*                                                                           */
/* Opérations de conversion entre 'float' et 'Float16' et 'BFloat16'.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FLOATCONVERSION_H
#define ARCCORE_BASE_FLOATCONVERSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <cstring>

// TODO: Utiliser 'std::endian' lorsqu'on sera en C++20.
// TODO: Utiliser 'std::bit_cast' au lieu de 'std::memcpy' pour rendre les fonctions 'constexpr'

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

// The following Float16_t conversions are based on the code from
// Eigen library.

// The conversion routines are Copyright (c) Fabian Giesen, 2016.
// The original license follows:
//
// Copyright (c) Fabian Giesen, 2016
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

namespace detail
{
  union float32_bits
  {
    unsigned int u;
    float f;
  };
} // namespace detail

inline constexpr uint16_t
convertFloat16ToUint16Impl(float v)
{
  detail::float32_bits f{};
  f.f = v;

  constexpr detail::float32_bits f32infty = { 255 << 23 };
  constexpr detail::float32_bits f16max = { (127 + 16) << 23 };
  constexpr detail::float32_bits denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
  constexpr unsigned int sign_mask = 0x80000000u;
  uint16_t val = static_cast<uint16_t>(0x0u);

  unsigned int sign = f.u & sign_mask;
  f.u ^= sign;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.u >= f16max.u) { // result is Inf or NaN (all exponent bits set)
    val = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
  }
  else { // (De)normalized number or zero
    if (f.u < (113 << 23)) { // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.f += denorm_magic.f;

      // and one integer subtract of the bias later, we have our final float!
      val = static_cast<uint16_t>(f.u - denorm_magic.u);
    }
    else {
      unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd

      // update exponent, rounding bias part 1
      // Equivalent to `f.u += ((unsigned int)(15 - 127) << 23) + 0xfff`, but
      // without arithmetic overflow.
      f.u += 0xc8000fffU;
      // rounding bias part 2
      f.u += mant_odd;
      // take the bits!
      val = static_cast<uint16_t>(f.u >> 13);
    }
  }

  val |= static_cast<uint16_t>(sign >> 16);
  return val;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline float
convertToFloat16Impl(uint16_t val)
{
  constexpr detail::float32_bits magic = { 113 << 23 };
  constexpr unsigned int shifted_exp = 0x7c00 << 13; // exponent mask after shift
  detail::float32_bits o{};

  o.u = (val & 0x7fff) << 13; // exponent/mantissa bits
  unsigned int exp = shifted_exp & o.u; // just the exponent
  o.u += (127 - 15) << 23; // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) { // Inf/NaN?
    o.u += (128 - 16) << 23; // extra exp adjust
  }
  else if (exp == 0) { // Zero/Denormal?
    o.u += 1 << 23; // extra exp adjust
    o.f -= magic.f; // re-normalize
  }

  // original code:
  o.u |= (val & 0x8000U) << 16U; // sign bit

  return o.f;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Converti en appliquant l'arrondi au plus proche (round-to-nearest)
inline uint16_t
convertBFloat16ToUint16Impl(float v)
{
  uint32_t input = 0;
  std::memcpy(&input, &v, sizeof(uint32_t));
  // Least significant bit of resulting bfloat.
  uint32_t lsb = (input >> 16) & 1;
  uint32_t rounding_bias = 0x7fff + lsb;
  input += rounding_bias;
  uint16_t output = static_cast<uint16_t>(input >> 16);
  return output;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline float
convertToBFloat16Impl(uint16_t val)
{
  float result;
  char* const first = reinterpret_cast<char*>(&result);
  char* const second = first + sizeof(uint16_t);
  // Les macros suivantes ne sont pas définies sous Windows mais ce dernier
  // ne supporte que des architectures big-endian donc cela ne pose pas
  // de problèmes.
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
  std::memcpy(first, &val, sizeof(uint16_t));
  std::memset(second, 0, sizeof(uint16_t));
#else
  std::memset(first, 0, sizeof(uint16_t));
  std::memcpy(second, &val, sizeof(uint16_t));
#endif
  return result;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

namespace Arccore::impl
{
using Arcane::impl::convertFloat16ToUint16Impl;
using Arcane::impl::convertToFloat16Impl;
using Arcane::impl::convertBFloat16ToUint16Impl;
using Arcane::impl::convertToBFloat16Impl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
