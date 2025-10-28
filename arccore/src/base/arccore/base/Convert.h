// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Fonctions pour convertir une chaîne de caractère en un type donné.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_CONVERT_H
#define ARCCORE_BASE_CONVERT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringView.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert::Impl
{
/*!
 * \brief Encapsule un std::istream pour un StringView.
 *
 * Actuellement (C++20) std::istringstream utilise en
 * entrée un std::string ce qui nécessite une instance de ce type
 * et donc une allocation potentielle. Cette classe sert à éviter
 * cela en utilisant directement la mémoire pointée par l'instance
 * de StringView passé dans le constructeur. Cette dernière doit
 * rester valide durant toute l'ulisation de cette classe.
 */
class ARCCORE_BASE_EXPORT StringViewInputStream
: private std::streambuf
{
 public:

  explicit StringViewInputStream(StringView v);

 public:

  std::istream& stream() { return m_stream; }

 private:

  StringView m_view;
  std::istream m_stream;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert::Impl

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
#ifdef ARCCORE_REAL_USE_APFLOAT
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
#ifdef ARCCORE_REAL_USE_APFLOAT
  return static_cast<Real>(static_cast<long>(r));
#else
  return static_cast<Real>(r);
#endif
}
//! Converti \c r en un \c Real
inline Real
toReal(unsigned long long r)
{
#ifdef ARCCORE_REAL_USE_APFLOAT
  return static_cast<Real>(static_cast<unsigned long>(r));
#else
  return static_cast<Real>(r);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

