// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Fonctions pour convertir un type en un autre.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CONVERT_H
#define ARCANE_UTILS_CONVERT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arccore/base/StringView.h"

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Real en \c double
inline double
toDouble(Real r)
{
#ifdef ARCANE_REAL_USE_APFLOAT
  return ap2double(r.ap);
#else
  return static_cast<double>(r);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Real en \c Integer
inline Integer
toInteger(Real r)
{
  return static_cast<Integer>(toDouble(r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Real en \c Int64
inline Int64
toInt64(Real r)
{
  return static_cast<Int64>(toDouble(r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Real en \c Int32
inline Int32
toInt32(Real r)
{
  return static_cast<Int32>(toDouble(r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Converti un \c Real en \c Integer
inline bool
toBool(Real r)
{
  return static_cast<bool>(toDouble(r));
}

//! Converti \c r en un \c Real
inline Real
toReal(Real r)
{
  return r;
}
//! Converti \c r en un \c Real
inline Real
toReal(int r)
{
  return static_cast<Real>(r);
}
//! Converti \c r en un \c Real
inline Real
toReal(unsigned int r)
{
  return static_cast<Real>(r);
}
//! Converti \c r en un \c Real
inline Real
toReal(long r)
{
  return static_cast<Real>(r);
}
//! Converti \c r en un \c Real
inline Real
toReal(unsigned long r)
{
  return static_cast<Real>(r);
}

//! Converti \c r en un \c Real
inline Real
toReal(long long r)
{
#ifdef ARCANE_REAL_USE_APFLOAT
  return static_cast<Real>(static_cast<long>(r));
#else
  return static_cast<Real>(r);
#endif
}
//! Converti \c r en un \c Real
inline Real
toReal(unsigned long long r)
{
#ifdef ARCANE_REAL_USE_APFLOAT
  return static_cast<Real>(static_cast<unsigned long>(r));
#else
  return static_cast<Real>(r);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un tableau d'octet en sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(ByteConstArrayView input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un tableau d'octet en sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(Span<const std::byte> input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un réel en sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(Real input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un entier 64 bits sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 * Le tableau \a output doit avoir au moins 16 éléments.
 */
extern ARCANE_UTILS_EXPORT void
toHexaString(Int64 input, Span<Byte> output);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class Type;

template <typename T>
class ScalarType
{
 public:

  //! Convertit \a s en le type \a T
  ARCANE_UTILS_EXPORT static std::optional<T> tryParse(StringView s);

  /*!
   * \brief Convertit \a s en le type \a T.
   *
   * Si \a s.empty() est vrai, alors retourne \a default_value.
   */
  static std::optional<T>
  tryParseIfNotEmpty(StringView s, const T& default_value)
  {
    return (s.empty()) ? default_value : tryParse(s);
  }

  /*!
   * \brief Convertit la valeur de la variable d'environnement \a s en le type \a T.
   *
   * Si platform::getEnvironmentVariable(s) est nul, return std::nullopt.
   * Sinon, retourne cette valeur convertie en le type \a T. Si la conversion
   * n'est pas possible, retourne std::nullopt si \a throw_if_invalid vaut \a false ou
   * lève une exception s'il vaut \a true.
   */
  ARCANE_UTILS_EXPORT static std::optional<T>
  tryParseFromEnvironment(StringView s, bool throw_if_invalid);
};

//! Spécialisation pour les types scalaires
template <> class Type<Int64> : public ScalarType<Int64>
{};
template <> class Type<Int32> : public ScalarType<Int32>
{};
template <> class Type<Real> : public ScalarType<Real>
{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

