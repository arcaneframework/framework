// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDynamicMachineMemoryWindowBaseInternal.h                   (C) 2000-2025 */
/*                                                                           */
/* Interface de classe permettant de créer des fenêtres mémoires pour un     */
/* noeud de calcul.                                                          */
/* Les segments de ces fenêtres ne sont pas contigüs en mémoire et peuvent   */
/* être redimensionnés.                                                      */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSING_INTERNAL_IDYNAMICMACHINEMEMORYWINDOWBASEINTERNAL_H
#define ARCCORE_MESSAGEPASSING_INTERNAL_IDYNAMICMACHINEMEMORYWINDOWBASEINTERNAL_H

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

/*!
 * \brief Classe permettant de créer des fenêtres mémoires pour un noeud de
 * calcul.
 *
 * Les segments de ces fenêtres ne seront pas contigüs en mémoire et pourront
 * être redimensionnés (une fenêtre par processus et un segment par fenêtre).
 *
 * La méthode add() pouvant vouloir redimensionner un segment, et ce
 * redimensionnement étant une opération collective, un appel à syncAdd() doit
 * être réalisé après les add().
 *
 * Afin d'avoir des add() non concurrents, cette opération est possible
 * uniquement sur le segment que nous possédons.
 * Pour être plus flexible, il est possible d'échanger ses segments, ce qui
 * permet de faire des add() sur le segment d'un autre processus sans
 * problème.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IDynamicMachineMemoryWindowBaseInternal
{
 public:

  virtual ~IDynamicMachineMemoryWindowBaseInternal() = default;

 public:

  /*!
   * \brief Méthode permettant d'obtenir la taille d'un élement de la fenêtre.
   *
   * Appel non collectif.
   *
   * \return La taille d'un élement.
   */
  virtual Int32 sizeofOneElem() const = 0;

/*!
   * \brief Méthode permettant d'obtenir une vue sur le segment que nous
   * possédons.
   *
   * Appel non collectif.
   *
   * Dans le cas d'échange de segments, il est possible d'obtenir le
   * propriétaire du segment que nous possédons avec la méthode segmentOwner().
   *
   * \return Une vue.
   */
  virtual Span<std::byte> segment() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment que possède un
   * autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * Dans le cas d'échange de segments, il est possible d'obtenir le
   * propriétaire du segment avec la méthode segmentOwner(rank).
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  virtual Span<std::byte> segment(Int32 rank) const = 0;

  /*!
   * \brief Méthode permettant d'obtenir le propriétaire du segment que nous
   * possédons.
   *
   * Appel non collectif.
   *
   * Méthode utile lors de l'échange de segments.
   *
   * \return Le propriétaire du segment.
   */
  virtual Int32 segmentOwner() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir le propriétaire du segment que
   * possède un autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * Méthode utile lors de l'échange de segments.
   *
   * \return Le propriétaire du segment.
   */
  virtual Int32 segmentOwner(Int32 rank) const = 0;

  /*!
   * \brief Méthode permettant d'ajouter un ou plusieurs éléments dans le
   * segment que nous possédons.
   *
   * L'appel à cette méthode n'est pas collectif mais peut nécessiter des
   * synchronisations si elle a besoin de redimensionner le segment.
   * Une barrier spécifique est disponible (syncAdd()) et doit être mise juste
   * après les add(). Les autres méthodes collectives ne doivent surtout pas
   * être appelées entre les add() et le syncAdd() (sinon deadlock).
   *
   * Chaque processus est libre de faire autant de add() qu'il veut avant le
   * syncAdd(). Pas besoin d'avoir le même nombre de add() par processus.
   *
   * \param elem Le ou les élements à ajouter.
   */
  virtual void add(Span<const std::byte> elem) = 0;

  /*!
   * \brief Méthode permettant d'échanger le segment que nous possédons avec
   * le segment de \a rank.
   *
   * Appel collectif.
   *
   * Cet échange permet de faire des add() sur le segment d'un autre processus
   * sans avoir besoin de protéger des accès concurrents.
   *
   * Pour effectuer un échange entre deux processus (disons 0 et 1), les deux
   * processus doivent appeler cette méthode avec le rang de l'autre :
   * - 0 doit appeler exchangeSegmentWith(1),
   * - 1 doit appeler exchangeSegmentWith(0).
   *
   * Les processus ne souhaitant pas échanger de segments doivent appeler
   * \a exchangeSegmentWith() (ou exchangeSegmentWith(2) pour le processus 2).
   *
   * Si nécessaire, il est possible de récupérer le rang d'un segment "loué"
   * avec les méthodes \a segmentOwner().
   *
   * \param rank Le rang avec qui échanger son segment.
   */
  virtual void exchangeSegmentWith(Int32 rank) = 0;
  /*!
   * Voir \a exchangeSegmentWith(Int32 rank).
   */
  virtual void exchangeSegmentWith() = 0;

  /*!
   * \brief Méthode permettant de réinitialiser les échanges effectués.
   *
   * Appel non collectif.
   *
   * Chaque processus retrouvera son segment.
   */
  virtual void resetExchanges() = 0;

  /*!
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * Appel non collectif.
   *
   * \return Une vue contenant les ids des rangs.
   */
  virtual ConstArrayView<Int32> machineRanks() const = 0;

  /*!
   * \brief Méthode permettant de synchroniser les add().
   *
   * Appel collectif.
   *
   * Cette méthode doit être appelée après le ou les add() du ou des
   * processus. Aucun appel collectif ne doit être fait entre les add() et
   * le syncAdd().
   */
  virtual void syncAdd() = 0;

  /*!
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  virtual void barrier() = 0;

  /*!
   * \brief Méthode permettant de réserver de l'espace mémoire dans le segment
   * que nous possédons.
   *
   * Appel collectif.
   *
   * Cette méthode ne fait rien si \a new_capacity est inférieur à l'espace
   * mémoire déjà alloué pour le segment.
   * Pour les processus ne souhaitant pas augmenter l'espace mémoire
   * disponible pour leur segment, la méthode \a reserve() (sans arguments)
   * est disponible.
   *
   * MPI réservera un espace avec une taille supérieur ou égale à
   * \a new_capacity.
   *
   * Cette méthode ne redimensionne pas le segment, il faudra toujours passer
   * par la méthode add() pour ajouter des éléments.
   *
   * Pour redimensionner le segment, la méthode \a resize(Int64 new_size) est
   * disponible.
   *
   * \param new_capacity La nouvelle capacité demandée.
   */
  virtual void reserve(Int64 new_capacity) = 0;

  /*!
   * Voir \a reserve(Int64 new_capacity)
   */
  virtual void reserve() = 0;

  /*!
   * \brief Méthode permettant de redimensionner le segment que nous
   * possédons.
   *
   * Appel collectif.
   *
   * Si la taille fournie est inférieure à la taille actuelle du segment, les
   * éléments situés après la taille fournie seront supprimés.
   *
   * Pour les processus ne souhaitant pas redimensionner leur segment, il est
   * possible d'appeler la méthode \a resize() (sans arguments).
   *
   * \param new_size La nouvelle taille.
   */
  virtual void resize(Int64 new_size) = 0;

  /*!
   * Voir \a resize(Int64 new_size)
   */
  virtual void resize() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
