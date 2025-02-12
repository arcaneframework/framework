// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayViewCommon.h                                           (C) 2000-2025 */
/*                                                                           */
/* Déclarations communes aux classes ArrayView, ConstArrayView et Span.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYVIEWCOMMON_H
#define ARCCORE_BASE_ARRAYVIEWCOMMON_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayIterator.h"

#include <iostream>

// 'assert' est nécessaire pour le code accélérateur
#include <assert.h>

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

//! Indique si les deux vues sont égales
template<typename View2DType> inline bool
areEqual2D(View2DType rhs, View2DType lhs)
{
  using size_type = typename View2DType::size_type;
  const size_type dim1_size = rhs.dim1Size();
  const size_type dim2_size = rhs.dim2Size();
  if (dim1_size!=lhs.dim1Size())
    return false;
  if (dim2_size!=lhs.dim2Size())
    return false;
  for( size_type i=0; i<dim1_size; ++i ){
    for( size_type j=0; j<dim2_size; ++j ){
      if (rhs(i,j)!=lhs(i,j))
        return false;
    }
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

//! Teste si \a size est positif ou nul et lance une exception si ce n'est pas le cas
inline constexpr ARCCORE_HOST_DEVICE void
arccoreCheckIsPositive(Int64 size)
{
  if (size<0){
#ifdef ARCCORE_DEVICE_CODE
    assert("'size' is negative");
#else
    impl::arccoreThrowNegativeSize(size);
#endif
  }
}

//! Teste si \a size est plus petit que ARCCORE_INTEGER_MAX et lance une exception si ce n'est pas le cas
inline constexpr ARCCORE_HOST_DEVICE void
arccoreCheckIsValidInteger(Int64 size)
{
  if (size>=ARCCORE_INTEGER_MAX){
#ifdef ARCCORE_DEVICE_CODE
    assert("'size' is bigger than ARCCORE_INTEGER_MAX");
#else
    impl::arccoreThrowTooBigInteger(size);
#endif
  }
}

//! Teste si \a size est plus petit que ARCCORE_INT64_MAX et lance une exception si ce n'est pas le cas
inline constexpr ARCCORE_HOST_DEVICE void
arccoreCheckIsValidInt64(size_t size)
{
  if (size>=ARCCORE_INT64_MAX){
#ifdef ARCCORE_DEVICE_CODE
    assert("'size' is bigger than ARCCORE_INT64_MAX");
#else
    impl::arccoreThrowTooBigInt64(size);
#endif
  }
}

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
inline constexpr ARCCORE_HOST_DEVICE Integer
arccoreCheckArraySize(unsigned long long size)
{
  impl::arccoreCheckIsValidInteger(size);
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
  impl::arccoreCheckIsValidInteger(size);
  impl::arccoreCheckIsPositive(size);
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
  impl::arccoreCheckIsValidInteger(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 *
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr ARCCORE_HOST_DEVICE Integer
arccoreCheckArraySize(long size)
{
  impl::arccoreCheckIsValidInteger(size);
  impl::arccoreCheckIsPositive(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr ARCCORE_HOST_DEVICE Integer
arccoreCheckArraySize(unsigned int size)
{
  impl::arccoreCheckIsValidInteger(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr ARCCORE_HOST_DEVICE Integer
arccoreCheckArraySize(int size)
{
  impl::arccoreCheckIsValidInteger(size);
  impl::arccoreCheckIsPositive(size);
  return static_cast<Integer>(size);
}

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Int64' pour servir
 * de taille à un tableau.
 *
 * Si possible, retourne \a size convertie en un 'Int64'. Sinon, lance
 * une exception de type ArgumentException.
 */
inline constexpr ARCCORE_HOST_DEVICE Int64
arccoreCheckLargeArraySize(size_t size)
{
  impl::arccoreCheckIsValidInt64(size);
  return static_cast<Int64>(size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IntType> class ArraySizeChecker;

//! Spécialisation pour tester la conversion en Int32
template <>
class ArraySizeChecker<Int32>
{
 public:

  template <typename SizeType> ARCCORE_HOST_DEVICE static Int32 check(SizeType size)
  {
    return arccoreCheckArraySize(size);
  }
};

//! Spécialisation pour tester la conversion en Int64
template <>
class ArraySizeChecker<Int64>
{
 public:

  static ARCCORE_HOST_DEVICE Int64 check(std::size_t size)
  {
    return arccoreCheckLargeArraySize(size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{
using Arccore::impl::arccoreCheckIsPositive;
using Arccore::impl::arccoreCheckIsValidInt64;
using Arccore::impl::arccoreCheckIsValidInteger;
using Arccore::impl::arccoreThrowNegativeSize;
using Arccore::impl::arccoreThrowTooBigInt64;
using Arccore::impl::arccoreThrowTooBigInteger;
using Arccore::impl::areEqual;
using Arccore::impl::areEqual2D;
using Arccore::impl::dumpArray;
using Arccore::impl::subViewInterval;
} // namespace Arcane::impl

namespace Arcane
{
using Arccore::arccoreCheckArraySize;
using Arccore::arccoreCheckLargeArraySize;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
