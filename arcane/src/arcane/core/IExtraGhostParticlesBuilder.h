// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExtraGhostParticlesBuilder.h                               (C) 2000-2022 */
/*                                                                           */
/* Interface d'un constructeur de mailles fantômes "extraordinaires"         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IEXTRAGHOSTPARTICLESBUILDER_H
#define ARCANE_IEXTRAGHOSTPARTICLESBUILDER_H
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
 * \brief Interface d'un constructeur de mailles fantômes "extraordinaires".
 *
 * Une maille fantôtme "extraordinaire" est une maille fantôme ajoutée aux
 * mailles fantôtmes définies par la connectivité du maillage. En particulier,
 * le calcul des mailles fantômes extraordinaires est effectué à chaque mise
 * à jour du maillage ou équilibrage de charge.
 *
 * \note rend obsolète le paramètre \a remove_old_ghost de la méthode IMesh::endUpdate().
 *
 */
class IExtraGhostParticlesBuilder
{
 public:
  
  virtual ~IExtraGhostParticlesBuilder() {} //!< Libère les ressources.
  
 public:

  /*!
   * \brief Calcul des mailles "extraordinaires" à envoyer.
   *
   * Effectue le calcul des mailles "extraordinaires" suivant
   * un algorithme de construction  
   */
  virtual void computeExtraParticlesToSend() =0;

  /*!
   * \brief Indices locaux des mailles "extraordinaires" pour envoi
   *
   * Récupère le tableau des mailles "extraordinaires" à destination
   * du sous-domaine \a rank
   */
  virtual Int32ConstArrayView extraParticlesToSend(const String& family_name,Int32 rank) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

