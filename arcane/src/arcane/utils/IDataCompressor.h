// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataCompressor.h                                           (C) 2000-2021 */
/*                                                                           */
/* Interface permettant de compresser/décompresser des données.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IDATACOMPRESSOR_H
#define ARCANE_UTILS_IDATACOMPRESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service permettant de compresser/décompresser des données.
 */
class ARCANE_UTILS_EXPORT IDataCompressor
{
 public:
  
  virtual ~IDataCompressor() = default;

 public:

  virtual void build() =0;

 public:

  //! Nom de l'algorithme
  virtual String name() const =0;

 /*!
  * \brief Taille minimale du tableau en dessous de laquelle il n'est pas utile
  * de compresser.
  *
  * Cela peut être utilisé par l'appelant pour ne pas pas compresser/décompresser
  * certains tableaux. Cette valeur n'est pas utilisée en interne par cette instance.
  *
  * Si l'appelant utilise cette valeur, il faut garantir la cohérence à la fois
  * en compression et décompression (i.e: ne pas appeler la décompression pour les
  * tableaux dont la taille décompressée est inférieure à minCompressSize() si
  * la méthode compress() n'a pas été appelée pour ce tableau.
  */
  virtual Int64 minCompressSize() const =0;

  /*!
   * \brief Compresse les données \a values et les stocke dans \a compressed_values.
   *
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  virtual void compress(Span<const std::byte> values,Array<std::byte>& compressed_values) =0;

  /*!
   * \brief Décompresse les données \a compressed_values et les stocke dans \a values.
   *
   * \a values doit déjà avoir été allouée à la taille nécessaire pour contenir
   * les données décompressées.
   * Cette opération peut lever une exception de type IOException en cas d'erreur.
   */
  virtual void decompress(Span<const std::byte> compressed_values,Span<std::byte> values) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
