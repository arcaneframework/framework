/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <arccore/base/ArrayView.h>
#include <arccore/collections/Array.h>

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Recherche d'un �l�ment dans un tableau
namespace ArrayScan
{
  //! Recherche lin�aire de la valeur \a x dans un tableau non-ordonn�e \a v avec contr�le
  //! de validit�
  /*! Ce type de recherche est performante sur des tableaux de petites tailles (effet
   * cache) Si plusieurs instance de la valeur \a x existe, la premi�re sera trouv�e.
   */
  template <typename T>
  inline Integer exhaustiveScan(const T& x, ConstArrayView<T> v);

  //! Recherche lin�aire de la valeur \a x dans un tableau ordonn�e \a v avec contr�le de
  //! validit�
  /*! Ce type de recherche est performante sur des tableaux de petites tailles (effet
   * cache) Si plusieurs instance de la valeur \a x existe, la premi�re sera trouv�e.
   */
  template <typename T>
  inline Integer linearScan(const T& x, ConstArrayView<T> v);
  //! Recherche dichotomique de la valeur \a x dans un tableau ordonn�e \a v avec contr�le
  //! de validit�
  /*! Cette recherche dichotomique est hybrid�e avec une recherche lin�aire par de
   * meilleures performances
   *  sur toutes tailles de tableau
   *  Si plusieurs instance de la valeur \a x existe, une quelconque occurence sera
   * trouv�e.
   */
  template <typename T>
  inline Integer dichotomicScan(const T& x, ConstArrayView<T> v);

  //! Recherche lin�aire de la position d'insertion de la valeur \a x dans un tableau
  //! ordonn�e \a v avec contr�le de validit�
  /*! Ce type de recherche est performante sur des tableaux de petites tailles (effet
   * cache) Si plusieurs instance de la valeur \a x existe, la premi�re sera trouv�e.
   */
  template <typename T>
  inline Integer linearPositionScan(const T& x, ConstArrayView<T> v);
  //! Recherche dichotomique de la  position d'insertion de la valeur \a x dans un tableau
  //! ordonn�e \a v avec contr�le de validit�
  /*! Cette recherche dichotomique est hybrid�e avec une recherche lin�aire par de
   * meilleures performances
   *  sur toutes tailles de tableau
   *  Si plusieurs instance de la valeur \a x existe, une quelconque occurence sera
   * trouv�e.
   */
  template <typename T>
  inline Integer dichotomicPositionScan(const T& x, ConstArrayView<T> v);

  //! Recherche lin�aire de la borne inf�rieur de l'intervalle contenant la valeur \a x
  //! dans un tableau ordonn�e \a v sans contr�le de validit�
  /*! L'absence de contr�le de validit� ne contr�le pas que le tableau est non vide et que
   *  l'�l�ment recherch� est dans l'intervalle d�fini par les extr�mit�s du tableau
   */
  template <typename T>
  inline Integer linearIntervalScan(const T& x, const Integer n, const T* vptr);

  //! Recherche dichotomique de la borne inf�rieur de l'intervalle contenant la valeur \a
  //! x dans un tableau ordonn�e \a v avec contr�le de validit�
  /*! Cette recherche dichotomique est hybrid� avec une recherche lin�aire par de
   * meilleures performances sur toutes tailles de tableau. L'absence de contr�le de
   * validit� ne contr�le pas que le tableau est non vide et que l'�l�ment recherch� est
   * dans l'intervalle d�fini par les extr�mit�s du tableau
   */
  template <typename T>
  inline Integer dichotomicIntervalScan(const T& x, const Integer n, const T* vptr);
} // namespace ArrayScan

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

namespace ArrayScan
{

  template <typename T>
  Integer exhaustiveScan(const T& x, ConstArrayView<T> v)
  {
    const Integer n = v.size();
    for (Integer i = 0; i < n; ++i)
      if (v[i] == x)
        return i;
    return -1;
  }

  template <typename T>
  Integer linearScan(const T& x, ConstArrayView<T> v)
  {
    const Integer n = v.size();
    Integer index = 0;
    while (index < n and v[index] < x) {
      ++index;
    }
    if (index == n or v[index] != x)
      return -1;
    else
      return index;
  }

  template <typename T>
  inline Integer dichotomicScan(const T& x, ConstArrayView<T> v)
  {
    const Integer n = v.size();

    Integer ileft = 0;
    Integer iright = n; // d�finit un intervale ouvert � droite
    static const Integer n_linear = 20; // threshold to switch from dichotomic to linear
    // scan \todo mettre cache line size

    // Start dichotomy
    if (n > n_linear) {
      if (x < v[0] or x > v[n - 1])
        return -1;
      do {
        const Integer imid = (iright + ileft) / 2;
        const T& vmid = v[imid];
        if (x < vmid) {
          iright = imid;
        }
        else if (x >= vmid) {
          ileft = imid;
        }
      } while (iright - ileft > n_linear);
    }

    // Switch to linear search
    // (exhaustiveScan en alternative; gain mitig�)
    while (ileft < iright and v[ileft] < x) {
      ++ileft;
    }
    if (ileft >= iright or v[ileft] != x)
      return -1;
    else
      return ileft;
  }

  /*---------------------------------------------------------------------------*/

  template <typename T>
  Integer linearPositionScan(const T& x, ConstArrayView<T> v)
  {
    const Integer n = v.size();
    Integer index = 0;
    while (index < n and v[index] < x) {
      ++index;
    }
    return index;
  }

  template <typename T>
  inline Integer dichotomicPositionScan(const T& x, ConstArrayView<T> v)
  {
    const Integer n = v.size();

    Integer ileft = 0;
    Integer iright = n; // d�finit un intervale ouvert � droite
    static const Integer n_linear = 20; // threshold to switch from dichotomic to linear
    // scan \todo mettre cache line size

    // Start dichotomy
    if (n > n_linear) {
      if (x < v[0])
        return 0;
      if (x > v[n - 1])
        return n;
      do {
        const Integer imid = (iright + ileft) / 2;
        const T& vmid = v[imid];
        if (x < vmid) {
          iright = imid;
        }
        else if (x >= vmid) {
          ileft = imid;
        }
      } while (iright - ileft > n_linear);
    }

    // Switch to linear search
    // (exhaustiveScan en alternative; gain mitig�)
    while (ileft < iright and v[ileft] < x) {
      ++ileft;
    }
    return ileft;
  }

  /*---------------------------------------------------------------------------*/

  template <typename T>
  Integer linearIntervalScan(const T& x, const Integer n, const T* vptr)
  {
    // Prepare
    Integer index = 0;

    // Search
    while (vptr[index + 1] <= x)
      ++index;
    return index;
  }

  template <typename T>
  Integer dichotomicIntervalScan(const T& x, const Integer n, const T* vptr)
  {
    //! Prepare

    Integer ileft = 0;
    Integer iright = n; // d�finit un intervalle ouvert � droite
    static const Integer n_linear = 20; // threshold to switch from dichotomic to linear
    // scan \todo mettre cache line size

    // Start dichotomy
    while (iright - ileft > n_linear) {
      const Integer imid = (iright + ileft) / 2;
      const T& vmid = vptr[imid];
      if (x < vmid) {
        iright = imid;
      }
      else {
        ileft = imid;
      }
    }

    // Switch to linear search
    while (vptr[ileft + 1] <= x)
      ++ileft;
    return ileft;
  }

} // end of namespace ArrayScan

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArrayConversion
{
} // end of namespace ArrayConversion

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
insert(UniqueArray<Integer>& list, UniqueArray<Real>& value, Integer entry, Real eps = 0.)
{
  if (entry == 0) {
    list[0] = 0;
    return;
  }
  Integer size = entry;
  Integer i = size;
  Real z = value[entry];
  for (Integer k = 0; k < size; k++) {
    if (z - value[list[k]] >= eps) {
      i = k;
      break;
    }
  }
  Integer last = entry;
  for (Integer j = i; j < size; j++) {
    Integer tmp = list[j];
    list[j] = last;
    last = tmp;
  }
  list[size] = last;
}

inline Real
average(ArrayView<Real> x, ArrayView<Real> coef, Integer n)
{
  Real xx = 0.;
  for (Integer i = 0; i < n; i++)
    xx += coef[i] * x[i];
  return xx;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
