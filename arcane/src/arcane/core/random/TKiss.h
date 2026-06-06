// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TKiss.h                                                     (C) 2000-2025 */
/*                                                                           */
/* This file defines the TKiss class template and the associated class Kiss. */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_TKISS_H
#define ARCANE_CORE_RANDOM_TKISS_H
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

/*! Kiss class template. It allows defining Kiss generator classes. The
 * pseudo-random numbers generated are of type UIntType. The generation of
 * these numbers is performed by calling the () operator. The state of the
 * generator is defined by a private member \c _state[i] of the class, which
 * is an array of five elements (0<i<=4). The seed (state \c state[i] 0<i<=4
 * initial of the generator, also called seed array) is initialized by
 * calling the constructor or the various \c seed methods available.
*/
template <typename UIntType, UIntType val>
class TKiss
{
 public:

  typedef UIntType result_type;
  typedef UIntType state_type;
  static const bool has_fixed_range = true;
  static const result_type min_value = 0;
  static const result_type max_value = 4294967295U;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Returns the minimum possible value of a sequence.
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  result_type min() const { return min_value; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Returns the maximum possible value of a sequence.
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  result_type max() const { return max_value; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Constructor with initialization of the seed array from the argument values.
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  explicit TKiss(UIntType x0 = 30903, UIntType y0 = 30903, UIntType z0 = 30903, UIntType w0 = 30903, UIntType carry0 = 0)
  {
    _state[0] = x0;
    _state[1] = y0;
    _state[2] = z0;
    _state[3] = w0;
    _state[4] = carry0;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Initialization of the seed array from the \c state.
   *          The generator state \c state must consist of five elements.
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  void seed(UIntType* state)
  {
    for (Integer i = 0; i < 5; i++)
      _state[i] = state[i];
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Initialization of the seed array from the value \c x0.
   *          The seed array of this generator consists of five elements.
   *          The first four elements take the value \c x0. The fifth
   *          element takes the zero value.
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  void seed(UIntType x0)
  {
    for (Integer i = 0; i < 4; i++)
      _state[i] = x0;
    _state[4] = 0;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Initialization of the seed array from the argument values.
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  void seed(UIntType x0, UIntType y0, UIntType z0, UIntType w0, UIntType carry0)
  {
    _state[0] = x0;
    _state[1] = y0;
    _state[2] = z0;
    _state[3] = w0;
    _state[4] = carry0;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Method that returns the i-th component of the generator state. The complete generator state is given by the indices \c i ranging between 0 and 4 ( 0 < \c i <= 4 ).
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  UIntType getState(Integer i) const { return _state[i]; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Overdefinition of the \c () operator which returns the pseudo-random value. The generator state is modified.
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  UIntType operator()()
  {
    UIntType t;
    _state[0] = _state[0] * 69069 + 1;
    _state[1] ^= _state[1] << 13;
    _state[1] ^= _state[1] >> 17;
    _state[1] ^= _state[1] << 5;

    t = (_state[3] << 1) + _state[2] + _state[4];
    _state[4] = ((_state[2] >> 2) + (_state[3] >> 3) + (_state[4] >> 2)) >> 30;
    _state[2] = _state[3];
    _state[3] = t;
    return (_state[0] + _state[1] + _state[2]);
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Validation function (I don't know what it's for!)
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  bool validation(UIntType x) const { return val == x; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \brief Overdefinition of the == operator
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  bool operator==(const TKiss& rhs) const
  {
    return (_state[0] == rhs._state[0]) && (_state[1] == rhs._state[1]) && (_state[2] == rhs._state[2]) && (_state[3] == rhs._state[3]) && (_state[4] == rhs._state[4]);
  }

 private:

  state_type _state[5];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef TKiss<UInt32, 0> Kiss;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::random

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
