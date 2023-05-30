// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.h                                                   (C) 2000-2022 */
/*                                                                           */
/* Fonctions pour convertir un type en un autre.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CONVERT_H
#define ARCANE_UTILS_CONVERT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fonctions pour convertir un type en un autre.
 */
namespace Convert
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
  return (double)r;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Real en \c Integer
inline Integer
toInteger(Real r)
{
  return (Integer)toDouble(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Real en \c Int64
inline Int64
toInt64(Real r)
{
  return (Int64)toDouble(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Real en \c Int32
inline Int32
toInt32(Real r)
{
  return (Int32)toDouble(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Real en \c Integer
inline bool
toBool(Real r)
{
  return (bool)toDouble(r);
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
  return (Real)r;
}
//! Converti \c r en un \c Real
inline Real
toReal(unsigned int r)
{
  return (Real)r;
}
//! Converti \c r en un \c Real
inline Real
toReal(long r)
{
  return (Real)r;
}
//! Converti \c r en un \c Real
inline Real
toReal(unsigned long r)
{
  return (Real)r;
}

//! Converti \c r en un \c Real
inline Real
toReal(long long r)
{
#ifdef ARCANE_REAL_USE_APFLOAT
  return (Real)((long)r);
#else
  return (Real)r;
#endif
}
//! Converti \c r en un \c Real
inline Real
toReal(unsigned long long r)
{
#ifdef ARCANE_REAL_USE_APFLOAT
  return (Real)((unsigned long)r);
#else
  return (Real)r;
#endif
}

/*!
 * \brief Converti un tableau d'octet en sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(ByteConstArrayView input);

/*!
 * \brief Converti un tableau d'octet en sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(Span<const std::byte> input);

/*!
 * \brief Converti un réel en sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(Real input);

/*!
 * \brief Converti un entier 64 bits sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 * Le tableau \a output doit avoir au moins 16 éléments.
 */
extern ARCANE_UTILS_EXPORT void
toHexaString(Int64 input,Span<Byte> output);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Convert

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

