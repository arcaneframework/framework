// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueChecker.h                                              (C) 2000-2024 */
/*                                                                           */
/* Vérification de la validité de certaines valeurs.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_VALUECHECKER_H
#define ARCANE_UTILS_VALUECHECKER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/TraceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérification de la validité de certaines valeurs.
 *
 * Cette classe fournit un ensemble de méthodes permettant de vérifier
 * que des valeurs sont conformes à une référence.
 *
 * Par défaut, si une valeur est différente de sa référence, une exception
 * est lancée. Il est possible de changer ce comportement en mettant
 * setThrowIfError() à \a false. Dans ce cas, il est possible de
 * lancer manuellement l'exception en appelant throwIfError().
 */
class ARCANE_UTILS_EXPORT ValueChecker
{
 public:

  ValueChecker(const TraceInfo& ti)
  : m_trace_info(ti)
  , m_nb_error(0)
  , m_throw_on_error(true)
  {}

 public:

  /*!
   * Vérifie que \a value et \a expected_value ont les mêmes valeurs.
   */
  template <typename T1, typename T2, typename X = std::is_convertible<T2, T1>>
  void areEqual(const T1& value, const T2& expected_value, const String& message)
  {
    if (value != expected_value) {
      _addError(String::format("{0} value={1} expected={2}", message, value, expected_value));
    }
  }

  /*!
   * \brief Vérifie que les deux tableaux \a values et \a expected_values
   * ont les mêmes valeurs.
   */
  template <typename T1, typename T2,
            typename ValueType = typename T1::value_type,
            typename X1 = std::is_convertible<T1, Span<const ValueType>>,
            typename X2 = std::is_convertible<T1, Span<const ValueType>>>
  void areEqualArray(const T1& x_values, const T2& x_expected_values,
                     const String& message)
  {
    auto values = static_cast<Span<const ValueType>>(x_values);
    auto expected_values = static_cast<Span<const ValueType>>(x_expected_values);

    Int64 nb_value = values.size();
    Int64 nb_expected = expected_values.size();
    if (nb_value != nb_expected) {
      _addError(String::format("{0} bad array size n={1} expected={2}",
                               message, nb_value, nb_expected));
      // Ne compare pas les éléments du tableau si les tailles
      // sont différentes.
      return;
    }

    for (Int64 i = 0; i < nb_value; ++i) {
      const ValueType& v = values[i];
      const ValueType& e = expected_values[i];
      if (v != e) {
        _addError(String::format("{0} index={1} value={2} expected={3}", message, i, v, e));
      }
    }
  }

  /*!
   * \brief Vérifie que les deux tableaux 2D \a values et \a expected_values
   * ont les mêmes valeurs.
   */
  template <typename T>
  void areEqualArray(Span2<const T> values, Span2<const T> expected_values,
                     const String& message)
  {
    Int64 nb_value = values.dim1Size();
    Int64 nb_expected = expected_values.dim1Size();
    if (nb_value != nb_expected) {
      _addError(String::format("{0} bad array size n={1} expected={2}",
                               message, nb_value, nb_expected));
      // Ne compare pas les éléments du tableau si les tailles
      // sont différentes.
      return;
    }

    for (Int64 i = 0; i < nb_value; ++i)
      areEqualArray(values[i], expected_values[i], message);
  }

  /*!
   * \brief Vérifie que les deux tableaux \a values et \a expected_values
   * ont les mêmes valeurs.
   */
  template <typename T>
  void areEqualArray(SmallSpan2<T> values, SmallSpan2<T> expected_values,
                     const String& message)
  {
    return areEqualArray(Span2<const T>(values), Span2<const T>(expected_values), message);
  }

  /*!
   * \brief Lève une exception si nbError()!=0.
   */
  void throwIfError();

  //! Indique si on lève une exception en cas d'erreur
  void setThrowOnError(bool v)
  {
    m_throw_on_error = v;
  }

  //! Indique si on lève une exception en cas d'erreur
  bool throwOnError() const { return m_throw_on_error; }

  //! Nombre d'erreurs
  Integer nbError() const { return m_nb_error; }

 private:

  TraceInfo m_trace_info;
  Integer m_nb_error;
  OStringStream m_ostr;
  String m_last_error_str;
  bool m_throw_on_error;

 private:

  void _addError(const String& message);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

