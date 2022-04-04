﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayViewCommon.h                                           (C) 2000-2021 */
/*                                                                           */
/* Déclarations communes aux classes ArrayView, ConstArrayView et Span.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYVIEWCOMMON_H
#define ARCCORE_BASE_ARRAYVIEWCOMMON_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayIterator.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Sous-vue correspondant à l'interval \a index sur \a nb_interval
template<typename ViewType> ARCCORE_HOST_DEVICE
auto subViewInterval(ViewType view,
                     typename ViewType::size_type index,
                     typename ViewType::size_type nb_interval) -> ViewType
{
  using size_type = typename ViewType::size_type;
  if (nb_interval<=0)
    return ViewType();
  if (index<0 || index>=nb_interval)
    return ViewType();
  size_type n = view.size();
  size_type isize = n / nb_interval;
  size_type ibegin = index * isize;
  // Pour le dernier interval, prend les elements restants
  if ((index+1)==nb_interval)
    isize = n - ibegin;
  return ViewType::create(view.data()+ibegin,isize);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Affiche les valeurs de la vue.
 *
 * Affiche sur le flot \a o les valeurs de \a val.
 * Si \a max_print est supérieur à 0, indique le nombre maximum de valeurs
 * à afficher.
 */
template<typename ViewType> inline void
dumpArray(std::ostream& o,ViewType val,int max_print)
{
  using size_type = typename ViewType::size_type;
  size_type n = val.size();
  if (max_print>0 && n>max_print){
    // N'affiche que les (max_print/2) premiers et les (max_print/2) derniers
    // sinon si le tableau est très grand cela peut générer des
    // sorties listings énormes.
    size_type z = (max_print/2);
    size_type z2 = n - z;
    o << "[0]=\"" << val[0] << '"';
    for( size_type i=1; i<z; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
    o << " ... ... (skipping indexes " << z << " to " << z2 << " ) ... ... ";
    for( size_type i=(z2+1); i<n; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
  }
  else{
    for( size_type i=0; i<n; ++i ){
      if (i!=0)
        o << ' ';
      o << "[" << i << "]=\"" << val[i] << '"';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si les deux vues sont égales
template<typename ViewType> inline bool
areEqual(ViewType rhs, ViewType lhs)
{
  using size_type = typename ViewType::size_type;
  if (rhs.size()!=lhs.size())
    return false;
  size_type s = rhs.size();
  for( size_type i=0; i<s; ++i ){
    if (rhs[i]!=lhs[i])
      return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Lance une exception 'ArgumentException'
extern "C++" ARCCORE_BASE_EXPORT void
arccoreThrowTooBigInteger [[noreturn]] (std::size_t size);

extern "C++" ARCCORE_BASE_EXPORT void
arccoreThrowTooBigInt64 [[noreturn]] (std::size_t size);

extern "C++" ARCCORE_BASE_EXPORT void
arccoreThrowNegativeSize [[noreturn]] (Int64 size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::impl

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr Integer
arccoreCheckArraySize(unsigned long long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    impl::arccoreThrowTooBigInteger(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr Integer
arccoreCheckArraySize(long long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    impl::arccoreThrowTooBigInteger(size);
  if (size<0)
    impl::arccoreThrowNegativeSize(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    impl::arccoreThrowTooBigInteger(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 *
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr Integer
arccoreCheckArraySize(long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    impl::arccoreThrowTooBigInteger(size);
  if (size<0)
    impl::arccoreThrowNegativeSize(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr Integer
arccoreCheckArraySize(unsigned int size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    impl::arccoreThrowTooBigInteger(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr Integer
arccoreCheckArraySize(int size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    impl::arccoreThrowTooBigInteger(size);
  if (size<0)
    impl::arccoreThrowNegativeSize(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Int64' pour servir
 * de taille à un tableau.
 *
 * Si possible, retourne \a size convertie en un 'Int64'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr Int64
arccoreCheckLargeArraySize(size_t size)
{
  if (size>=ARCCORE_INT64_MAX)
    impl::arccoreThrowTooBigInt64(size);
  return static_cast<Int64>(size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename IntType> class ArraySizeChecker;

//! Spécialisation pour tester la conversion en Int32
template<>
class ArraySizeChecker<Int32>
{
 public:
  template<typename SizeType>
  static Int32 check(SizeType size)
  {
    return arccoreCheckArraySize(size);
  }
};

//! Spécialisation pour tester la conversion en Int64
template<>
class ArraySizeChecker<Int64>
{
 public:
  static Int64 check(std::size_t size)
  {
    return arccoreCheckLargeArraySize(size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
