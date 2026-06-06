// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRandomNumberGenerator.h                                    (C) 2000-2022 */
/*                                                                           */
/* Interface for random number generator.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_IRANDOMNUMBERGENERATOR_H
#define ARCANE_IRANDOMNUMBERGENERATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Class allowing easy manipulation of a seed.
 * 
 * A seed is represented by an array of Bytes.
 * This class uses an ArrayView of this array.
 * 
 * This class allows defining a value in the array and
 * retrieving that value (other things).
 * 
 * This class does not store the array but only
 * an ArrayView of this array.
 * 
 */
class ARCANE_CORE_EXPORT RNGSeedHelper
{
 public:

  /**
   * @brief Class constructor.
   * 
   * @param av An ArrayView of an array representing a seed.
   */
  RNGSeedHelper(ByteArrayView av)
  {
    m_seed = av;
  }

  /**
   * @brief Class constructor.
   * 
   * @tparam T A base type.
   * @param var A pointer to the seed 
   *            (note, does not make a copy of the value!).
   */
  template <class T>
  RNGSeedHelper(T* var)
  {
    m_seed = ByteArrayView(sizeof(T), (Byte*)var);
  }

  virtual ~RNGSeedHelper() = default;

 public:

  /**
   * @brief Method allowing setting a value in the seed.
   * 
   * @tparam T The value type.
   * @param value_in The future value of the seed.
   * @return true If the value could be assigned.
   * @return false If the value could not be assigned.
   */
  template <class T>
  bool setValue(T value_in)
  {
    if (m_seed.empty()) {
      return false;
    }
    memcpy(m_seed.data(), &value_in, std::min(m_seed.size(), (Integer)sizeof(T)));
    for (Integer i = sizeof(T); i < m_seed.size(); i++) {
      m_seed[i] = 0x00;
    }
    return true;
  }

  /**
   * @brief Method allowing retrieval of the seed value.
   * 
   * @tparam T The seed type.
   * @param value_out [OUT] The seed value.
   * @param without_size_check If value truncation is allowed.
   * @return true If the value could be retrieved.
   * @return false If the value could not be retrieved or if the array 
   *               has a null size.
   */
  template <class T>
  bool value(T& value_out, bool without_size_check = true) const
  {
    if (m_seed.empty() || (!without_size_check && sizeof(T) != m_seed.size())) {
      return false;
    }
    value_out = 0;
    memcpy(&value_out, m_seed.data(), std::min(m_seed.size(), (Integer)sizeof(T)));
    return true;
  }

  /**
   * @brief Method allowing retrieval of the seed value.
   * 
   * @tparam T The seed type.
   * @param value_out [OUT] The seed value.
   * @param without_size_check If value truncation is allowed.
   * @return true If the value could be retrieved.
   * @return false If the value could not be retrieved or if the array 
   *               has a null size.
   */
  template <class T>
  bool value(T* value_out, bool without_size_check = true) const
  {
    if (m_seed.empty() || (sizeof(T) != m_seed.size() && !without_size_check)) {
      return false;
    }
    *value_out = 0;
    memcpy(value_out, m_seed.data(), std::min(m_seed.size(), (Integer)sizeof(T)));
    return true;
  }

  /**
   * @brief Method allowing retrieval of the seed size.
   * 
   * @return Integer The seed size (in bytes).
   */
  Integer sizeOfSeed() const
  {
    return m_seed.size();
  }

  /**
   * @brief Method allowing retrieval of a constant view.
   * 
   * @return ByteConstArrayView The view.
   */
  ByteConstArrayView constView() const
  {
    return m_seed.constView();
  }

  /**
   * @brief Method allowing retrieval of a view.
   * 
   * @return ByteArrayView The view.
   */
  ByteArrayView view()
  {
    return m_seed;
  }

  /**
   * @brief Copy operator from a seed value.
   * 
   * @tparam T The seed type.
   * @param value The seed value.
   * @return RNGSeedHelper& The destination seed.
   */
  template <class T>
  RNGSeedHelper& operator=(T new_value)
  {
    setValue(new_value);
    return *this;
  }

  /**
   * @brief Method allowing retrieval of a copy of the Byte array.
   * 
   * @return ByteUniqueArray The copy of the Byte array.
   */
  ByteUniqueArray copy()
  {
    return ByteUniqueArray(m_seed);
  }

 protected:

  ByteArrayView m_seed;
};

/**
 * @ingroup StandardService
 * @brief Interface for a random number generator.
 */
class ARCANE_CORE_EXPORT IRandomNumberGenerator
{
 public:

  virtual ~IRandomNumberGenerator() = default;

 public:

  /**
   * @brief Method allowing initialization of the service.
   * 
   * With the seed optional (or the default seed if in singleton mode).
   * 
   * @return true If initialization was successful.
   * @return false If initialization did not occur.
   */
  virtual bool initSeed() = 0;

  /**
   * @brief Method allowing initialization of the service.
   * 
   * If the seed does not have the correct size, false will be returned.
   * 
   * @param seed The original seed.
   * @return true If initialization was successful.
   * @return false If initialization did not occur.
   */
  virtual bool initSeed(ByteArrayView seed) = 0;

  /**
   * @brief Method allowing retrieval of a constant view of the current seed.
   * 
   * @return ByteArrayView The seed.
   */
  virtual ByteConstArrayView viewSeed() = 0;

  /**
   * @brief Method allowing retrieval of an empty seed of the correct size.
   * 
   * @return ByteUniqueArray The empty seed.
   */
  virtual ByteUniqueArray emptySeed() = 0;

  /**
   * @brief Method allowing knowledge of the seed size required
   * for the implementation.
   * 
   * @return Integer The required seed size (in bytes).
   */
  virtual Integer neededSizeOfSeed() = 0;

  /**
   * @brief Method allowing knowledge if leaps are allowed on the
   * seed generator.
   * 
   * @return true If yes.
   * @return false If no.
   */
  virtual bool isLeapSeedSupported() = 0;

  /**
   * @brief Method allowing generation of a "child" seed from a "parent" seed.
   * 
   * @param leap The leap to perform (0 = seed n+1+0 / 1 = seed n+1+1).
   * @return ByteUniqueArray The new seed generated from the seed in memory.
   */
  virtual ByteUniqueArray generateRandomSeed(Integer leap = 0) = 0;

  /**
   * @brief Method allowing generation of a "child" seed from a "parent" seed.
   * 
   * This method does not use the seed in memory but the seed provided as a parameter.
   * If the seed provided as a parameter does not have the correct size, an error will be raised.
   * 
   * @param parent_seed The "parent" seed.
   * @param leap The leap to perform (0 = seed n+1+0 / 1 = seed n+1+1).
   * @return ByteUniqueArray The new seed generated from the "parent" seed.
   */
  virtual ByteUniqueArray generateRandomSeed(ByteArrayView parent_seed, Integer leap = 0) = 0;

  /**
   * @brief Method allowing knowledge if leaps are allowed on the
   * number generator.
   * 
   * @return true If yes.
   * @return false If no.
   */
  virtual bool isLeapNumberSupported() = 0;

  /**
   * @brief Method allowing generation of a random number using the seed in memory.
   * 
   * @param leap The leap to perform (0 = number n+1+0 / 1 = number n+1+1).
   * @return Real The generated number (between 0 and 1).
   */
  virtual Real generateRandomNumber(Integer leap = 0) = 0;

  /**
   * @brief Method allowing generation of a random number using the seed passed as a parameter.
   *    
   * This method does not use the seed in memory but the seed provided as a parameter.
   * If the seed provided as a parameter does not have the correct size, an error will be raised.
   * 
   * @param seed The seed.
   * @param leap The leap to perform (0 = number n+1+0 / 1 = number n+1+1).
   * @return Real The generated number (between 0 and 1).
   */
  virtual Real generateRandomNumber(ByteArrayView seed, Integer leap = 0) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
