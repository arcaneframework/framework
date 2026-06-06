// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* InversiveCongruential.h                                     (C) 2000-2025 */
/*                                                                           */
/* This file defines the InversiveCongruential class pattern as well as a    */
/* associated class Hellekalek1995. It is an adapted version of the file     */
/* InversiveCongruential.hpp from the BOOST library */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_INVERSIVECONGRUENTIAL_H
#define ARCANE_CORE_RANDOM_INVERSIVECONGRUENTIAL_H
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

/*! Class pattern \c InversiveCongruential. It allows defining classes
 * of the Inversive Congruential type based on the parameters \c a,
 * \c c, and \c m. The generated pseudo-random numbers are of type \c IntType.
 *
 * The generation of a sequence of pseudo-random numbers is done:
 *
 * - either by successive calls to the operator \c (). In this case, the seed can  
 *   be initialized by calling the constructor or the different methods  
 *   \c seed. The generator state is managed internally via the  
 *   private member \c _x. Its value is accessible via the \c getState() method.
 *
 * - or by calling the method \c apply(x). The generator state \c x 
 *   is managed outside the class. The \c seed and \c getState() methods 
 *   are meaningless in this usage.
*/
template<typename IntType, IntType a, IntType c, IntType m, IntType val>
class InversiveCongruential
{
 public:
  typedef IntType result_type;
  static const bool has_fixed_range = true;
  static const result_type min_value = ( c == 0 ? 1 : 0 );
  static const result_type max_value = m-1;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Returns the minimum possible value of a sequence.
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  result_type min() const { return c == 0 ? 1 : 0; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Returns the maximum possible value of a sequence.
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  result_type max() const { return m-1; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Constructor with seed initialization from the value
   *         \c x0.
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */  
  explicit InversiveCongruential(IntType x0 = 1)
    : _x(x0)
  { 
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Initialization of the generator seed from the value \c x0.
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  void seed(IntType x0) { _x = x0; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Method that returns the generator state.
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date   28/07/2006
   */
  IntType getState() const { return _x; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Overriding the operator () which returns the pseudo
   * random value of the generator. The generator state is modified. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  IntType operator()()
  {
    _x = apply(_x);
    return _x;
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Returns the pseudo-random value from the state \c x. The
 * private member \c _x of the generator is not used and is not  
 * modified. 
 *
 * \author Patrick Rathouit (origine bibliotheque BOOST)
 * \date  28/07/2006
 */
  static IntType apply(IntType x)
  {
    typedef utils::const_mod<IntType, m> do_mod;
    return x = do_mod::mult_add(a,do_mod::invert(x), c);
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Validation function (I don't really know what it is for!)
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  bool validation(IntType x) const { return val == x; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Overriding the == operator
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  bool operator==(const InversiveCongruential& rhs) const
    { return _x == rhs._x; }

 private:

  IntType _x;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef InversiveCongruential<Int32, 9102, 2147483647-36884165,
  2147483647, 0> Hellekalek1995;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
