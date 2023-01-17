// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDeflateService.h                                           (C) 2000-2020 */
/*                                                                           */
/* Interface d'un service permettant de compresser/décompresser des données. */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDEFLATESERVICE_H
#define ARCANE_IDEFLATESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Taille minimale (en octet) pour qu'un tableau soit compressé.
// En dessous de cette taille il n'est pas compressé.
// Si cette taille change, il faut changer la taille correspondante
// dans le VariableComparer en C#.
static const Integer DEFLATE_MIN_SIZE = 512;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service permettant de compresser/décompresser des données.
 */
class ARCANE_CORE_EXPORT IDeflateService
{
 public:
  
  virtual ~IDeflateService() = default;

 public:

  virtual void build() =0;

 public:

  /*!
   * \brief Compresse les données \a values et les stocke dans \a compressed_values.
   *
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  virtual void compress(ByteConstArrayView values,ByteArray& compressed_values) =0;

  /*!
   * \brief Compresse les données \a values et les stocke dans \a compressed_values.
   *
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  virtual void compress(Span<const Byte> values,ByteArray& compressed_values);

  /*!
   * \brief Décompresse les données \a compressed_values et les stocke dans \a values.
   *
   * \a values doit déjà avoir été alloué à la taille nécessaire pour contenir
   * les données décompressées.
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  virtual void decompress(ByteConstArrayView compressed_values,ByteArrayView values) =0;

  /*!
   * \brief Décompresse les données \a compressed_values et les stocke dans \a values.
   *
   * \a values doit déjà avoir été alloué à la taille nécessaire pour contenir
   * les données décompressées.
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  virtual void decompress(Span<const Byte> compressed_values,Span<Byte> values);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
