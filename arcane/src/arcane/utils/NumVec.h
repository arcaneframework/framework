// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumVec.h                                                    (C) 2000-2022 */
/*                                                                           */
/* Vecteur de taille fixe de types numériques.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMVEC_H
#define ARCANE_UTILS_NUMVEC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Petit vecteur de taille fixe de N données numériques.
 *
 * \note Actuellement uniquement implémenté pour 2 ou 3 valeurs.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane
 */
template <typename T, int Size>
class NumVec
{
  static_assert(Size == 2 || Size == 3, "Valid values for Size are 2 or 3");

 public:

  using ThatClass = NumVec<T, Size>;
  using DataType = T;

 public:

  //! Construit le vecteur nul.
  NumVec() = default;

  //! Construit avec le couple (ax,ay)
  template <int S = Size, typename = std::enable_if_t<S == 2, void>>
  constexpr ARCCORE_HOST_DEVICE NumVec(T ax, T ay)
  {
    m_values[0] = ax;
    m_values[1] = ay;
  }

  //! Construit avec le triplet (ax,ay,az)
  template <int S = Size, typename = std::enable_if_t<S == 3, void>>
  constexpr ARCCORE_HOST_DEVICE NumVec(T ax, T ay, T az)
  {
    m_values[0] = ax;
    m_values[1] = ay;
    m_values[2] = az;
  }

  //! Construit l'instance avec pour chaque composante la valeur \a v
  explicit constexpr ARCCORE_HOST_DEVICE NumVec(std::array<T, Size> v)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = v[i];
  }

  //! Construit l'instance avec pour chaque composante la valeur \a v
  explicit constexpr ARCCORE_HOST_DEVICE NumVec(T v)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = v;
  }

  template <int S = Size, typename = std::enable_if_t<S == 2, void>>
  explicit constexpr ARCCORE_HOST_DEVICE NumVec(Real2 v)
  : NumVec(v.x, v.y)
  {}

  template <int S = Size, typename = std::enable_if_t<S == 3, void>>
  explicit constexpr ARCCORE_HOST_DEVICE NumVec(Real3 v)
  : NumVec(v.x, v.y, v.z)
  {}

  //! Affecte à l'instance le triplet (v,v,v).
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(Real v)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] = v;
    return (*this);
  }

  template <int S = Size, typename = std::enable_if_t<S == 2, void>>
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(const Real2& v)
  {
    *this = ThatClass(v);
    return (*this);
  }

  template <int S = Size, typename = std::enable_if_t<S == 3, void>>
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(const Real3& v)
  {
    *this = ThatClass(v);
    return (*this);
  }

  template <int S = Size, typename = std::enable_if_t<S == 2, void>>
  operator Real2() const { return Real2(m_values[0], m_values[1]); }

  template <int S = Size, typename = std::enable_if_t<S == 3, void>>
  operator Real3() const { return Real3(m_values[0], m_values[1], m_values[2]); }

 public:

  constexpr ARCCORE_HOST_DEVICE static ThatClass zero() { return ThatClass(); }

 public:

  constexpr ARCCORE_HOST_DEVICE bool isNearlyZero() const
  {
    bool is_nearly_zero = true;
    for (int i = 0; i < Size; ++i)
      is_nearly_zero = is_nearly_zero && math::isNearlyZero(m_values[i]);
    return is_nearly_zero;
  }

  //! Retourne la norme L2 au carré du triplet \f$x^2+y^2+z^2\f$
  constexpr ARCCORE_HOST_DEVICE Real squareNormL2() const
  {
    T v = T();
    for (int i = 0; i < Size; ++i)
      v += m_values[i] * m_values[i];
    return v;
  }
  //! Retourne la norme L2 du triplet \f$\sqrt{x^2+y^2+z^2}\f$
  ARCCORE_HOST_DEVICE Real normL2() const { return _sqrt(squareNormL2()); }

  //! Valeur absolue composante par composante.
  ARCCORE_HOST_DEVICE ThatClass absolute() const
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = math::abs(m_values[i]);
    return v;
  }

  //std::ostream& print(std::ostream& o) const;
  //! Ecrit le triplet sur le flot \a o sous la forme (x,y,z)
  //std::ostream& printXyz(std::ostream& o) const;

  //! Ajoute \a b à chaque composante de l'instance
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator+=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] += b;
    return (*this);
  }

  //! Ajoute \a b à l'instance
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator+=(const ThatClass& b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] += b.m_values[i];
    return (*this);
  }
  //! Soustrait \a b à chaque composante de l'instance
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator-=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] -= b;
    return (*this);
  }
  //! Soustrait \a b à l'instance
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator-=(const ThatClass& b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] -= b.m_values[i];
    return (*this);
  }
  //! Multiple chaque composante par \a b
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator*=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] *= b;
    return (*this);
  }
  //! Divise chaque composante par \a b
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator/=(T b)
  {
    for (int i = 0; i < Size; ++i)
      m_values[i] /= b;
    return (*this);
  }
  //! Créé un triplet qui vaut ce triplet ajouté à \a b
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator+(const ThatClass& a, const ThatClass& b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a.m_values[i] + b.m_values[i];
    return v;
  }
  //! Créé un triplet qui vaut \a b soustrait de ce triplet
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator-(const ThatClass& a, const ThatClass& b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a.m_values[i] - b.m_values[i];
    return v;
  }
  //! Créé un triplet opposé au triplet actuel
  constexpr ARCCORE_HOST_DEVICE ThatClass operator-() const
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = -m_values[i];
    return v;
  }
  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(T a, const ThatClass& vec)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = a * vec.m_values[i];
    return v;
  }
  //! Multiplication par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator*(const ThatClass& vec, T b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = vec.m_values[i] * b;
    return v;
  }
  //! Division par un scalaire.
  friend constexpr ARCCORE_HOST_DEVICE ThatClass operator/(const ThatClass& vec, T b)
  {
    ThatClass v;
    for (int i = 0; i < Size; ++i)
      v.m_values[i] = vec.m_values[i] / b;
    return v;
  }

  /*!
   * \brief Compare composant pas composante l'instance courante à \a b.
   *
   * \retval true si this.x==b.x et this.y==b.y et this.z==b.z.
   * \retval false sinon.
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator==(const ThatClass& a, const ThatClass& b)
  {
    for (int i = 0; i < Size; ++i)
      if (!_eq(a.m_values[i], b.m_values[i]))
        return false;
    return true;
  }

  /*!
   * \brief Compare deux vecteurs
   * Pour la notion d'égalité, voir operator==()
   */
  friend constexpr ARCCORE_HOST_DEVICE bool operator!=(const ThatClass& a, const ThatClass& b)
  {
    return !(a == b);
  }

  constexpr ARCCORE_HOST_DEVICE T& operator()(Int32 i)
  {
    ARCCORE_CHECK_AT(i, Size);
    return m_values[i];
  }
  constexpr ARCCORE_HOST_DEVICE T operator()(Int32 i) const
  {
    ARCCORE_CHECK_AT(i, Size);
    return m_values[i];
  }

  template <int S = Size, typename = std::enable_if_t<S >= 1, void>>
  T& x()
  {
    return m_values[0];
  }
  template <int S = Size, typename = std::enable_if_t<S >= 1, void>>
  T x() const
  {
    return m_values[0];
  }

  template <int S = Size, typename = std::enable_if_t<S >= 2, void>>
  T& y()
  {
    return m_values[1];
  }
  template <int S = Size, typename = std::enable_if_t<S >= 2, void>>
  T y() const
  {
    return m_values[1];
  }

  template <int S = Size, typename = std::enable_if_t<S >= 3, void>>
  T& z()
  {
    return m_values[2];
  }
  template <int S = Size, typename = std::enable_if_t<S >= 3, void>>
  T z() const
  {
    return m_values[2];
  }

 private:

  //! Valeurs du vecteur
  T m_values[Size] = {};

 private:

  /*!
   * \brief Compare les valeurs de \a a et \a b avec le comparateur TypeEqualT
   * \retval true si \a a et \a b sont égaux,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE static bool
  _eq(T a, T b)
  {
    return math::isEqual(a, b);
  }
  //! Retourne la racine carrée de \a a
  ARCCORE_HOST_DEVICE static T _sqrt(T a)
  {
    return math::sqrt(a);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
