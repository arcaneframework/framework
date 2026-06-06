// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MersenneTwister.h                                           (C) 2000-2025 */
/*                                                                           */
/* This file defines the MersenneTwister class pattern as well as two        */
/* associated classes mt19937 and mt11213b. It is a version adapted for      */
/* TROLL of the MersenneTwister.hpp file from the BOOST library              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_MERSENNETWISTER_H
#define ARCANE_CORE_RANDOM_MERSENNETWISTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/random/RandomGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::random
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief MersenneTwister class pattern.
 *
 * It allows defining classes of 
 * Mersenne Twister type generators based on the parameters w,n,m,r,a,u
 * s,b,t,c and l. The generated pseudo-random numbers are of type UIntType.
 * The generation of these numbers is done by calling the \c () operator. The state
 * of the generator is defined by a private member x[] of the class, which is a
 * array of 2*n dimensions. The seed (initial state of the generator) can be
 * initialized by calling the constructors or the various \c seed
 * methods available.
*/
template <class UIntType, Integer w, Integer n, Integer m, Integer r, UIntType a, Integer u,
          Integer s, UIntType b, Integer t, UIntType c, Integer l, UIntType val>
class MersenneTwister
{
 public:

  typedef UIntType result_type;
  static const Integer word_size = w;
  static const Integer state_size = n;
  static const Integer shift_size = m;
  static const Integer mask_bits = r;
  static const UIntType parameter_a = a;
  static const Integer output_u = u;
  static const Integer output_s = s;
  static const UIntType output_b = b;
  static const Integer output_t = t;
  static const UIntType output_c = c;
  static const Integer output_l = l;

  static const bool has_fixed_range = false;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Constructor with seed initialization from the
   * seed() method
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date 28/07/2006
   */
  MersenneTwister() { seed(); }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Constructor with initialization of the seed array from
   * the value \c value. The call to the \c seed(value) method is performed.
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date 28/07/2006
   */
  explicit MersenneTwister(UIntType value)
  {
    seed(value);
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Constructor with initialization of the seed array from
   * the \c seed(first,last) method.
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date   28/07/2006
   */
  template <class It> MersenneTwister(It& first, It last) { seed(first, last); }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Constructor with initialization of the seed array from the
   * generator \c gen. \c gen must contain the () operator which must
   * return a value of type UIntType.
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date   28/07/2006
   */
  template <class Generator>
  explicit MersenneTwister(Generator& gen) { seed(gen); }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Initialization of the seed array. The call to the \c
   * seed(5489) method is performed.
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date   28/07/2006
   */
  void seed() { seed(UIntType(5489)); }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Initialization of the seed array from the value \c value.
   * The seed array of this generator consists of n elements.
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date   28/07/2006
   */
  void seed(UIntType value)
  {
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html
    const UIntType mask = ~0u;
    x[0] = value & mask;
    for (i = 1; i < n; i++) {
      // See Knuth "The Art of Computer Programming" Vol. 2, 3rd ed., page 106
      x[i] = (1812433253UL * (x[i - 1] ^ (x[i - 1] >> (w - 2))) + i) & mask;
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Initialization of the seed array from the array \c state.
   * \c state must be an array of n elements.
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  void seed(UIntType* state)
  {
    for (Integer i = 0; i < n; i++)
      x[i] = state[i];
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Initialization of the seed array from the generator \c gen.
   * \c gen is a class that must contain the () operator returning
   * a value of type UIntType.
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date 28/07/2006
   */
  template <class Generator>
  void seed(Generator& gen)
  {
    // For GCC, moving this function out-of-line prevents inlining, which may
    // reduce overall object code size.  However, MSVC does not grok
    // out-of-line definitions of member function templates.
    for (Integer j = 0; j < n; j++)
      x[j] = gen();
    i = n;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Method that returns the generator state for index \c j. The complete state
   * of the generator is given by the values of index \c j
   * between 0 and n (0 < \c j <= n)
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  UIntType getState(Integer j)
  {
    return x[j];
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief min() returns the minimum possible value of a sequence.
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date 28/07/2006
   */
  result_type min() const { return 0; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief max() returns the maximum possible value of a sequence.
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date 28/07/2006
   */
  result_type max() const
  {
    // avoid "left shift count >= with of type" warning
    result_type res = 0;
    for (Integer i = 0; i < w; ++i)
      res |= (1u << i);
    return res;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /* declaration of the () operator */
  result_type operator()();

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Validation function (I don't really know what it is for!)
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date 28/07/2006
   */
  static bool validation(result_type v) { return val == v; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Overriding the == operator
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date 28/07/2006
   */
  bool operator==(const MersenneTwister& rhs) const
  {
    // Use a member function; Streamable concept not supported.
    for (Integer j = 0; j < state_size; ++j)
      if (compute(j) != rhs.compute(j))
        return false;
    return true;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Overriding the != operator
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date 28/07/2006
   */
  bool operator!=(const MersenneTwister& rhs) const
  {
    return !(*this == rhs);
  }

 private:

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Private method that returns the generator state for index
   * \c index.
   *
   * \author Patrick Rathouit (origin BOOST library)
   * \date   28/07/2006
   */
  // returns x(i-n+index), where index is in 0..n-1
  UIntType compute(UIntType index) const
  {
    // equivalent to (i-n+index) % 2n, but doesn't produce negative numbers
    return x[(i + n + index) % (2 * n)];
  }
  void twist(Integer block);

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /* Class pattern members */
  // state representation: next output is o(x(i))
  //   x[0]  ... x[k] x[k+1] ... x[n-1]     x[n]     ... x[2*n-1]   represents
  //  x(i-k) ... x(i) x(i+1) ... x(i-k+n-1) x(i-k-n) ... x[i(i-k-1)]
  // The goal is to always have x(i-n) ... x(i-1) available for
  // operator== and save/restore.
  UIntType x[2 * n];
  Integer i;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Implementation of the "twist" operation associated with the Mersenne Twister.
 * The generator state is modified.
 *
 * \author Patrick Rathouit (origin BOOST library)
 * \date   28/07/2006
 */
template <class UIntType, Integer w, Integer n, Integer m, Integer r, UIntType a, Integer u,
          Integer s, UIntType b, Integer t, UIntType c, Integer l, UIntType val>
void MersenneTwister<UIntType, w, n, m, r, a, u, s, b, t, c, l, val>::twist(int block)
{
  const UIntType upper_mask = (~0u) << r;
  const UIntType lower_mask = ~upper_mask;

  if (block == 0) {
    for (Integer j = n; j < 2 * n; j++) {
      UIntType y = (x[j - n] & upper_mask) | (x[j - (n - 1)] & lower_mask);
      x[j] = x[j - (n - m)] ^ (y >> 1) ^ (y & 1 ? a : 0);
    }
  }
  else if (block == 1) {
    // split loop to avoid costly modulo operations
    { // extra scope for MSVC brokenness w.r.t. for scope
      for (Integer j = 0; j < n - m; j++) {
        UIntType y = (x[j + n] & upper_mask) | (x[j + n + 1] & lower_mask);
        x[j] = x[j + n + m] ^ (y >> 1) ^ (y & 1 ? a : 0);
      }
    }

    for (Integer j = n - m; j < n - 1; j++) {
      UIntType y = (x[j + n] & upper_mask) | (x[j + n + 1] & lower_mask);
      x[j] = x[j - (n - m)] ^ (y >> 1) ^ (y & 1 ? a : 0);
    }
    // last iteration
    UIntType y = (x[2 * n - 1] & upper_mask) | (x[0] & lower_mask);
    x[n - 1] = x[m - 1] ^ (y >> 1) ^ (y & 1 ? a : 0);
    i = 0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Overriding the () operator which returns the pseudo
 * random value of the generator. The generator state is modified.
 *
 * \author Patrick Rathouit (origin BOOST library)
 * \date   28/07/2006
 */
template <class UIntType, Integer w, Integer n, Integer m, Integer r, UIntType a, Integer u,
          Integer s, UIntType b, Integer t, UIntType c, Integer l, UIntType val>
inline typename MersenneTwister<UIntType, w, n, m, r, a, u, s, b, t, c, l, val>::result_type
MersenneTwister<UIntType, w, n, m, r, a, u, s, b, t, c, l, val>::operator()()
{
  if (i == n)
    twist(0);
  else if (i >= 2 * n)
    twist(1);
  // Step 4
  UIntType z = x[i];
  ++i;
  z ^= (z >> u);
  z ^= ((z << s) & b);
  z ^= ((z << t) & c);
  z ^= (z >> l);
  return z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/* Definition of the mt11213b class*/
typedef MersenneTwister<UInt32, 32, 351, 175, 19, 0xccab8ee7, 11,
                        7, 0x31b6ab00, 15, 0xffe50000, 17, 0xa37d3c92>
Mt11213b;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/* Definition of the Mt19937 class*/
typedef MersenneTwister<UInt32, 32, 624, 397, 31, 0x9908b0df, 11,
                        7, 0x9d2c5680, 15, 0xefc60000, 18, 3346425566U>
Mt19937;

} // namespace Arcane::random

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
