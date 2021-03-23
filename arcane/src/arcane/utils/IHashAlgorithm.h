// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IHashAlgorithm.h                                            (C) 2000-2020 */
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
	
  virtual ~IHashAlgorithm(){}

 public:

  /*!
   * \brief Calcule la fonction de hashage sur le tableau \a input.
   *
   * La fonction de hashage est <strong>ajoutée</string> dans \a output.
   * La longueur dépend de l'algorithme utilisé.
   */
  virtual void computeHash(ByteConstArrayView input,ByteArray& output) =0;

  /*!
   * \brief Calcule la fonction de hashage sur le tableau \a input.
   *
   * La fonction de hashage est <strong>ajoutée</string> dans \a output.
   * La longueur dépend de l'algorithme utilisé.
   */
  virtual void computeHash64(Span<const Byte> input,ByteArray& output)
  {
    computeHash(input.smallView(),output);
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namesapce Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

