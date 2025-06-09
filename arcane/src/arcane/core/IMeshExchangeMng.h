// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshExchangeMng.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire des échanges de maillages entre sous-domaines.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHEXCHANGEMNG_H
#define ARCANE_CORE_IMESHEXCHANGEMNG_H
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
 * \brief Interface du gestionnaire des échanges de maillages entre
 * sous-domaines.
 */
class ARCANE_CORE_EXPORT IMeshExchangeMng
{
 public:

  virtual ~IMeshExchangeMng() = default; //!< Libère les ressources

 public:

  //! Maillage associé
  virtual IPrimaryMesh* mesh() const = 0;

  /*!
   * \brief Débute un échange.
   *
   * \pre exchanger()==nullptr.
   *
   * Lorsqu'un échange est en cours, il est interdit de faire certaines
   * opérations sur le maillage comme par exemple créer une nouvelle famille
   * où ajouter des groupes.
   */
  virtual IMeshExchanger* beginExchange() = 0;

  /*!
   * \brief Signale que l'échange est terminé.
   *
   * Cela permet de désallouer les structures associées à l'échange.
   * \post exchanger()==nullptr.
   */
  virtual void endExchange() = 0;

  /*!
   * \brief Échangeur courant.
   *
   * L'échangeur est non nul que si on est entre un beginExchange() et un endExchange()
   */
  virtual IMeshExchanger* exchanger() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
