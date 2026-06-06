// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Mrg32k3a.h                                                  (C) 2000-2025 */
/*                                                                           */
/* This file defines the class template TMrg32k3a as well as the associated  */
/* class Mrg32k3a.                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_MRG32K3A_H
#define ARCANE_CORE_RANDOM_MRG32K3A_H
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

/*! Class template TMrg32k3a. It allows defining classes of 
 * Mrg32k3a type generators. The pseudo-random numbers generated are of 
 * type RealType. The generator state is characterized by six values of type 
 * RealType and can be managed internally by the private member _state[i] where 0<i<=5.
 *
 * The generation of a sequence of pseudo-random numbers is performed:
 *
 * - either by successive calls to the operator (). In this case, the seed can 
 *   be initialized by the different seed methods or when calling the 
 *   constructor. The generator state is managed internally via
 *   the private member _state[i] (0<i<=5). Its components i are accessible 
 *   via the getState(i) method.
 *
 * - or by calling the \c apply(value) method. The generator state is 
 *   managed outside the class. The \c seed and \c getState methods are meaningless 
 *   in this usage.
*/
template <typename RealType, Int32 val>
class TMrg32k3a
{
 public:

  typedef RealType result_type;
  typedef RealType state_type;
  static const bool has_fixed_range = true;
  static const Int32 min_value = 0;
  static const Int32 max_value = 1;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Constructor initializing the seed array from
   * the value \c x0. The \c seed(x0) method is called.
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  explicit TMrg32k3a(Int32 x0 = 1)
  {
    seed(x0);
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Constructor initializing the seed array from
   * the \c state array. \c state must be an array of six elements.
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  explicit TMrg32k3a(state_type* state)
  {
    for (Integer i = 0; i < 6; i++)
      _state[i] = state[i];
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Initialization of the seed array from the value \c x0.
   *          The seed array of this generator consists of six elements.
   *
   * \author  Patrick Rathouit
   * \date    28/07/2006
   */
  void seed(Int32 x0)
  {
    x0 = (x0 | 1);
    _state[0] = (state_type)x0;
    _state[1] = _state[0];
    _state[2] = _state[1];
    _state[3] = _state[2];
    _state[4] = _state[3];
    _state[5] = _state[4];
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Method that returns the generator state for index \c i. The complete
   * state of the generator is given by the index values \c i ranging
   * between 0 and 5 ( 0 < \c i <=5 ).
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  RealType getState(Integer i) const { return _state[i]; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Overloading of the \c () operator which returns the pseudo
   * random value of the generator. The generator state is modified.
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  RealType operator()()
  {
    RealType _x;
    _x = apply(_state);
    return _x;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Returns the pseudo-random value from the \c state.
   * The generator state state must consist of six elements.
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  static RealType apply(state_type* state)
  {
    long k;
    Real p;
    p = 1403580.0 * state[1] - 810728.0 * state[0];
    k = static_cast<long>(p / 4294967087.0);
    p -= k * 4294967087.0;
    if (p < 0.0)
      p += 4294967087.0;
    state[0] = state[1];
    state[1] = state[2];
    state[2] = p;

    p = 527612.0 * state[5] - 1370589.0 * state[3];
    k = static_cast<long>(p / 4294944443.0);
    p -= k * 4294944443.0;
    if (p < 0.0)
      p += 4294944443.0;
    state[3] = state[4];
    state[4] = state[5];
    state[5] = p;

    if (state[2] <= state[5])
      return ((state[2] - state[5] + 4294967087.0) / 4294967087.0);
    else
      return ((state[2] - state[5]) / 4294967087.0);
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Returns the minimum possible value of a sequence.
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  result_type min() const { return static_cast<result_type>(min_value); }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Returns the maximum possible value of a sequence.
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  result_type max() const { return static_cast<result_type>(max_value); }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Validation function (I'm not sure what it's for!)
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  bool validation(RealType x) const { return val == x; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Overloading of the == operator
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  bool operator==(const TMrg32k3a& rhs) const
  {
    return (_state[0] == rhs._state[0]) && (_state[1] == rhs._state[1]) && (_state[2] == rhs._state[2]) && (_state[3] == rhs._state[3]) && (_state[4] == rhs._state[4]) && (_state[5] == rhs._state[5]);
  }

 private:

  state_type _state[6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef TMrg32k3a<Real, 0> Mrg32k3a;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::random

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
