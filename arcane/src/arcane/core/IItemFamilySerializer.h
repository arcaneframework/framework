// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilySerializer.h                                     (C) 2011-2024 */
/*                                                                           */
/* Gère la sérialisation/désérialisation des entités d'une famille.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYSERIALIZER_H
#define ARCANE_CORE_IITEMFAMILYSERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère la sérialisation/désérialisation des entités d'une famille.
 */
class ARCANE_CORE_EXPORT IItemFamilySerializer
{
 public:

  virtual ~IItemFamilySerializer() = default;

 public:

  /*!
   * \brief Sérialise dans \a buf les entités de la famille \a family().
   *
   * En mode 'Put' ou 'Reserve', \a items contient les numéros locaux des mailles.
   * En mode 'Get', appelle \a deserializeItems() et \a items est inutilisé.
   */
  virtual void serializeItems(ISerializer* buf, Int32ConstArrayView items) = 0;

  /*!
   * \brief Désérialise depuis \a buf les entités de la famille \a family().
   *
   * Si \a items_lid n'est pas nul, contient en retour les numéros locaux
   * des mailles désérialisées.
   */
  virtual void deserializeItems(ISerializer* buf, Int32Array* items_lid) = 0;

  /*!
   * \brief Sérialise dans \a buf les relations des entités de la famille \a family().
   *
   * En mode 'Put' ou 'Reserve', \a items contient les numéros locaux des mailles.
   * En mode 'Get', appelle \a deserializeItemRelations() et \a items est inutilisé.
   */
  virtual void serializeItemRelations(ISerializer* buf, Int32ConstArrayView items) = 0;

  /*!
   * \brief Désérialise les relations depuis \a buf les entités de la famille \a family().
   *
   * Si \a items_lid n'est pas nul, contient en retour les numéros locaux
   * des mailles dont les relations ont été désérialisées.
   */
  virtual void deserializeItemRelations(ISerializer* buf, Int32Array* items_lid) = 0;

  //! Famille associée
  virtual IItemFamily* family() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

