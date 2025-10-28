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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

