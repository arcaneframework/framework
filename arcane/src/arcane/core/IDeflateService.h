// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDeflateService.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service permettant de compresser/décompresser des données. */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDEFLATESERVICE_H
#define ARCANE_CORE_IDEFLATESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

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
 *
 * \deprecated Utiliser IDataCompressor à la place
 */
class ARCANE_CORE_EXPORT IDeflateService
{
 public:

  virtual ~IDeflateService() = default;

 public:

  virtual void build() = 0;

 public:

  /*!
   * \brief Compresse les données \a values et les stocke dans \a compressed_values.
   *
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This interface is deprecated. Use IDataCompressor instead")
  virtual void compress(ByteConstArrayView values, ByteArray& compressed_values) = 0;

  /*!
   * \brief Compresse les données \a values et les stocke dans \a compressed_values.
   *
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This interface is deprecated. Use IDataCompressor instead")
  virtual void compress(Span<const Byte> values, ByteArray& compressed_values);

  /*!
   * \brief Décompresse les données \a compressed_values et les stocke dans \a values.
   *
   * \a values doit déjà avoir été alloué à la taille nécessaire pour contenir
   * les données décompressées.
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This interface is deprecated. Use IDataCompressor instead")
  virtual void decompress(ByteConstArrayView compressed_values, ByteArrayView values) = 0;

  /*!
   * \brief Décompresse les données \a compressed_values et les stocke dans \a values.
   *
   * \a values doit déjà avoir été alloué à la taille nécessaire pour contenir
   * les données décompressées.
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This interface is deprecated. Use IDataCompressor instead")
  virtual void decompress(Span<const Byte> compressed_values, Span<Byte> values);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
