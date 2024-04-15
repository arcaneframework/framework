// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashFunction.h                                              (C) 2000-2024 */
/*                                                                           */
/* Fonction de hachage.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_HASHFUNCTION_H
#define ARCANE_UTILS_HASHFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fonctor pour une fonction de hachage.
 */
template <class Type>
class IntegerHashFunctionT
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fonction de hashage pour un entier 32 bits.
 
 Thomas Wang (http://www.cris.com/~Ttwang/tech/inthash.htm)
*/
template <>
class IntegerHashFunctionT<Int32>
{
 public:

  static constexpr ARCCORE_HOST_DEVICE Int32 hashfunc(Int32 key)
  {
    key += ~(key << 15);
    key ^= (key >> 10);
    key += (key << 3);
    key ^= (key >> 6);
    key += ~(key << 11);
    key ^= (key >> 16);
    return key;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fonction de hashage pour un entier 64 bits.
 
 Thomas Wang (http://www.cris.com/~Ttwang/tech/inthash.htm)
*/
template <>
class IntegerHashFunctionT<Int64>
{
 public:

  static constexpr ARCCORE_HOST_DEVICE Int64 hashfunc(Int64 key)
  {
    key += ~(key << 32);
    key ^= (key >> 22);
    key += ~(key << 13);
    key ^= (key >> 8);
    key += (key << 3);
    key ^= (key >> 15);
    key += ~(key << 27);
    key ^= (key >> 31);
    return key;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fonction de hashage une chaîne de caractères.
 */
template <>
class IntegerHashFunctionT<StringView>
{
 public:

  ARCANE_UTILS_EXPORT static Int64 hashfunc(StringView str);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
template <typename KeyType>
class HashTraitsT
{
 public:

  typedef const KeyType& KeyTypeConstRef;
  typedef KeyType& KeyTypeRef;
  typedef KeyType KeyTypeValue;
  typedef KeyType HashValueType;
  typedef FalseType Printable;

 public:

  static HashValueType hashFunction(KeyTypeConstRef key);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Spécialisation pour les Int32
 */
template <>
class HashTraitsT<Int32>
{
 public:

  typedef Int32 KeyTypeConstRef;
  typedef Int32& KeyTypeRef;
  typedef Int32 KeyTypeValue;
  typedef TrueType Printable;
  typedef Int32 HashValueType;

 public:

  static constexpr ARCCORE_HOST_DEVICE Int32 hashFunction(Int32 key)
  {
    return IntegerHashFunctionT<Int32>::hashfunc(key);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Spécialisation pour les Int64
 */
template <>
class HashTraitsT<Int64>
{
 public:

  typedef Int64 KeyTypeConstRef;
  typedef Int64& KeyTypeRef;
  typedef Int64 KeyTypeValue;
  typedef Int64 HashValueType;
  typedef TrueType Printable;

 public:

  static constexpr ARCCORE_HOST_DEVICE Int64 hashFunction(Int64 key)
  {
    return IntegerHashFunctionT<Int64>::hashfunc(key);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
