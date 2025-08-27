// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshCompactMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire des compactages de familles d'un maillage.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHCOMPACTMNG_H
#define ARCANE_CORE_IMESHCOMPACTMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshCompacter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des compactages de familles d'un maillage.
 *
 * Lorsqu'un compactage est en cours, il est interdit de faire certaines
 * opérations sur le maillage comme par exemple créer une nouvelle famille
 * ou ajouter des groupes.
 *
 * Le début d'un compactage se fait via l'appel à beginCompact(). Une fois
 * le compactage terminé, il faut appeler endCompact() pour détruire
 * l'instance de IMeshCompacter.
 *
 * Pour plus d'informations sur le compactage, se reporter à la documentation
 * de IMeshCompacter.
 */
class ARCANE_CORE_EXPORT IMeshCompactMng
{
 public:

  virtual ~IMeshCompactMng() {} //<! Libère les ressources

 public:

  //! Maillage associé
  virtual IMesh* mesh() const =0;

  /*!
   * \brief Débute un compactage sur toutes les familles du maillage.
   *
   * \pre compacter()==nullptr.
   */
  virtual IMeshCompacter* beginCompact() =0;

  /*!
   * \brief Débute un compactage pour la famille d'entité \a family
   *
   * \pre compacter()==nullptr.
   *
   */
  virtual IMeshCompacter* beginCompact(IItemFamily* family) =0;

  /*!
   * \brief Signale que le compactage est terminé.
   *
   * Cela permet de désallouer les structures associées au compactage.
   * \post exchanger()==nullptr.
   */
  virtual void endCompact() =0;

  /*!
   * \brief Compacteur actif courant.
   *
   * Le compacteur est non nul que si on est entre un beginCompact()
   * et un endCompact()
   */
  virtual IMeshCompacter* compacter() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
