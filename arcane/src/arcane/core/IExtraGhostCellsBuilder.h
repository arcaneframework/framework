// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExtraGhostCellsBuilder.h                                   (C) 2000-2025 */
/*                                                                           */
/* Interface d'un constructeur de mailles fantômes "extraordinaires"         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IEXTRAGHOSTCELLSBUILDER_H
#define ARCANE_CORE_IEXTRAGHOSTCELLSBUILDER_H
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
 * \brief Interface d'un constructeur de mailles fantômes "extraordinaires"  
 *
 * Une maille fantôme "extraordinaire" est une maille fantôme ajoutée aux
 * mailles fantômes définies par la connectivité du maillage. En particulier,
 * le calcul des mailles fantômes extraordinaires est effectué à chaque mise
 * à jour du maillage ou équilibrage de charge.
 *
 * \note Rend obsolète le paramètre \a remove_old_ghost de la méthode IMesh::endUpdate().
 */
class IExtraGhostCellsBuilder
{
 public:
  
  virtual ~IExtraGhostCellsBuilder() {} //!< Libère les ressources.
  
 public:

  /*!
   * \brief Calcul des mailles "extraordinaires" à envoyer.
   *
   * Effectue le calcul des mailles "extraordinaires" suivant
   * un algorithme de construction  
   */
  virtual void computeExtraCellsToSend() =0;

  /*!
   * \brief Indices locaux des mailles "extraordinaires" pour envoi.
   *
   * Récupère le tableau des mailles "extraordinaires" à destination
   * du sous-domaine \a sid
   */
  virtual Int32ConstArrayView extraCellsToSend(Int32 rank) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

