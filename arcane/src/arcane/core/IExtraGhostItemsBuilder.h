// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExtraGhostItemsBuilder.h                                   (C) 2000-2025 */
/*                                                                           */
/* Comment on file content.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IEXTRAGHOSTITEMSBUILDER_H_
#define ARCANE_CORE_IEXTRAGHOSTITEMSBUILDER_H_
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un constructeur d'item fantômes "extraordinaires"
 *
 * Un item fantôme "extraordinaire" est un item fantôme ajoutée aux
 * item fantômes définies par la connectivité du maillage. En particulier,
 * le calcul des items fantômes extraordinaires est effectué à chaque mise
 * à jour du maillage ou équilibrage de charge. Cette interface est en particulier
 * utilisé pour les degrés de liberté.
 *
 * NB : rend obsolète le paramètre remove_old_ghost de la méthode endUpdate de IMesh
 *
 */
class ARCANE_CORE_EXPORT IExtraGhostItemsBuilder
{
 public:

  /** Destructeur de la classe */
  virtual ~IExtraGhostItemsBuilder() = default;

 public:

  /*!
   * \brief Calcul des items "extraordinaires" à envoyer
   * Effectue le calcul des items "extraordinaires" suivant
   * un algorithme de construction
   */
  virtual void computeExtraItemsToSend() = 0;

  /*!
   * \brief Indices locaux des items "extraordinaires" pour envoi
   * Récupère le tableau des items "extraordinaires" à destination
   * du sous-domaine \a sid
   */
  virtual ConstArrayView<Int32> extraItemsToSend(Int32 sid) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* IEXTRAGHOSTITEMSBUILDER_H_ */
