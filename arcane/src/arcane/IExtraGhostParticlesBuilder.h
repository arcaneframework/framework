// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExtraGhostParticlesBuilder.h                                   (C) 2000-2011 */
/*                                                                           */
/* Interface d'un constructeur de mailles fant?mes "extraordinaires"         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IEXTRAGHOSTPARTICLESBUILDER_H
#define ARCANE_IEXTRAGHOSTPARTICLESBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un constructeur de mailles fant?mes "extraordinaires"  
 *
 * Une maille fant?me "extraordinaire" est une maille fant?me ajout?e aux
 * mailles fant?mes d?finies par la connectivit? du maillage. En particulier,
 * le calcul des mailles fant?mes extraordinaires est effectu? ? chaque mise
 * ? jour du maillage ou ?quilibrage de charge.
 *
 * NB : rend obsol?te le param?tre remove_old_ghost de la m?thode endUpdate de IMesh
 *
 */
class IExtraGhostParticlesBuilder
{
public:
  
  virtual ~IExtraGhostParticlesBuilder() {} //!< Lib?re les ressources.
  
public:

  /*!
   * \brief Calcul des mailles "extraordinaires" ? envoyer
   * Effectue le calcul des mailles "extraordinaires" suivant
   * un algorithme de construction  
   */
  virtual void computeExtraParticlesToSend() =0;

  /*!
   * \brief Indices locaux des mailles "extraordinaires" pour envoi
   * R?cup?re le tableau des mailles "extraordinaires" ? destination
   * du sous-domaine \a sid
   */
  virtual IntegerConstArrayView extraParticlesToSend(String const& family_name,Integer sid) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

