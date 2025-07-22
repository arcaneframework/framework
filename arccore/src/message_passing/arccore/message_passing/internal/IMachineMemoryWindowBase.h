// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMachineMemoryWindowBase.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface de classe permettant de créer une fenêtre mémoire pour un noeud */
/* de calcul. Cette fenêtre sera contigüe en mémoire.                        */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSING_INTERNAL_IMACHINEMEMORYWINDOWBASE_H
#define ARCCORE_MESSAGEPASSING_INTERNAL_IMACHINEMEMORYWINDOWBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO AH : Voir pour ajouter la possibilité de faire une fenêtre non-contigüe.

/*!
 * \brief Classe permettant de créer une fenêtre mémoire pour un noeud
 * de calcul.
 *
 * Cette fenêtre sera contigüe en mémoire et sera accessible par
 * tous les processus du noeud.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMachineMemoryWindowBase
{
 public:

  virtual ~IMachineMemoryWindowBase() = default;

 public:

  /*!
   * \brief Méthode permettant d'obtenir la taille d'un élement de la fenêtre.
   *
   * \return La taille d'un élement.
   */
  virtual Int32 sizeofOneElem() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur son segment.
   *
   * \return Une vue.
   */
  virtual Span<std::byte> segment() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment d'un autre
   * sous-domaine du noeud.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  virtual Span<std::byte> segment(Int32 rank) const = 0;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur toute la fenêtre.
   *
   * \return Une vue.
   */
  virtual Span<std::byte> window() const = 0;

  /*!
   * \brief Méthode permettant de redimensionner les segments de la fenêtre.
   *
   * Appel collectif.
   *
   * La taille totale de la fenêtre doit être inférieure ou égale à la taille
   * d'origine.
   *
   * \param new_sizeof_segment La nouvelle taille de notre segment (en octet).
   */
  virtual void resizeSegment(Int64 new_sizeof_segment) = 0;

  /*!
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * L'ordre des processus de la vue retournée correspond à l'ordre des
   * segments dans la fenêtre.
   *
   * \return Une vue contenant les ids des rangs.
   */
  virtual ConstArrayView<Int32> machineRanks() const = 0;

  /*!
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  virtual void barrier() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

