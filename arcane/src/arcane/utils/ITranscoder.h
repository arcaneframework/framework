// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITranscoder.h                                               (C) 2000-2005 */
/*                                                                           */
/* Interface d'un convertisseur de/vers le format UTF-16.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ITRANSCODER_H
#define ARCANE_UTILS_ITRANSCODER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un convertisseur de/vers le format UTF-16.
 */
class ITranscoder
{
 public:

  virtual ~ITranscoder() {} //<! Libère les ressources

 public:

  virtual void build() =0;

 public:

  /*! \brief Traduit la source \a src de longueur \a src_len au format UTF-16
   *
   * Stocke la conversion dans \a out qui doit être préallablement alloué
   * et de longueur suffisante.
   */
  virtual void transcodeToUtf16(const Byte* src,Integer src_len,UChar* out) =0;

  /*! \brief Traduit la source \a src de longueur \a src_len depuis le format UTF-16
   *
   * Stocke la conversion dans \a out qui doit être préallablement alloué
   * et de longueur suffisante.
   */
  virtual void transcodeFromUtf16(const UChar* src,Integer src_len,Byte* out) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

