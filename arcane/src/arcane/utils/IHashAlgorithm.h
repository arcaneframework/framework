// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IHashAlgorithm.h                                            (C) 2000-2023 */
/*                                                                           */
/* Interface d'un algorithme de hashage.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IHASHALGORITHM_H
#define ARCANE_UTILS_IHASHALGORITHM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un algorithme de hashage.
 */
class ARCANE_UTILS_EXPORT IHashAlgorithm
{
 public:

  virtual ~IHashAlgorithm() = default;

 public:

  //NOTE: pour l'instant (version 3.10) par encore virtuel pure pour rester
  // compatible avec l'existant
  //! Nom de l'algorithme
  virtual String name() const;

  /*!
   * \brief Calcule la fonction de hashage sur le tableau \a input.
   *
   * La fonction de hashage est <strong>ajoutée</string> dans \a output.
   * La longueur dépend de l'algorithme utilisé.
   */
  virtual void computeHash64(Span<const Byte> input, ByteArray& output);

  /*!
   * \brief Calcule la fonction de hashage sur le tableau \a input.
   *
   * La fonction de hashage est <strong>ajoutée</string> dans \a output.
   * La longueur dépend de l'algorithme utilisé.
   */
  virtual void computeHash64(Span<const std::byte> input, ByteArray& output);

 public:

  /*!
   * \brief Calcule la fonction de hashage sur le tableau \a input.
   *
   * La fonction de hashage est <strong>ajoutée</string> dans \a output.
   * La longueur dépend de l'algorithme utilisé.
   */
  ARCANE_DEPRECATED_REASON("Y2023: Use computeHash64(Span<const std::byte> input,ByteArray& output) instead")
  virtual void computeHash(ByteConstArrayView input, ByteArray& output) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

