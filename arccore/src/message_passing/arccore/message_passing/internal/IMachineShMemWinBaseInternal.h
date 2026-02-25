// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMachineShMemWinBaseInternal.h                              (C) 2000-2026 */
/*                                                                           */
/* Interface de classe permettant de créer des fenêtres mémoires pour un     */
/* noeud de calcul.                                                          */
/* Les segments de ces fenêtres ne sont pas contigüs en mémoire et peuvent   */
/* être redimensionnés.                                                      */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSING_INTERNAL_IMACHINESHMEMWINBASEINTERNAL_H
#define ARCCORE_MESSAGEPASSING_INTERNAL_IMACHINESHMEMWINBASEINTERNAL_H

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
 * redimensionnement étant une opération collective, l'appel à add() est donc
 * une opération collective
 *
 * Afin d'avoir des add() non concurrents, cette opération est possible
 * uniquement sur notre segment.
 * Pour ajouter des éléments dans le segment d'un autre sous-domaine,
 * les méthodes addToAnotherSegment() sont disponibles.
 *
 * Toutes les tailles utilisées sont en octet. \a sizeof_type est utilisé
 * seulement par MPI (si utilisé) et à des fins de vérification.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMachineShMemWinBaseInternal
{
 public:

  virtual ~IMachineShMemWinBaseInternal() = default;

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
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * Appel non collectif.
   *
   * \return Une vue contenant les ids des rangs.
   */
  virtual ConstArrayView<Int32> machineRanks() const = 0;

  /*!
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  virtual void barrier() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur notre segment.
   *
   * Appel non collectif.
   *
   * \return Une vue.
   */
  virtual Span<std::byte> segmentView() = 0;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment d'un
   * autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  virtual Span<std::byte> segmentView(Int32 rank) = 0;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur notre segment
   *
   * Appel non collectif.
   *
   * \return Une vue.
   */
  virtual Span<const std::byte> segmentConstView() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment d'un
   * autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  virtual Span<const std::byte> segmentConstView(Int32 rank) const = 0;

  /*!
   * \brief Méthode permettant d'ajouter des élements dans notre segment.
   *
   * Appel collectif.
   *
   * \note Ne pas mélanger les appels de cette méthode avec les appels à
   * addToAnotherSegment().
   *
   * Si le segment est trop petit, il sera redimensionné.
   *
   * Les sous-domaines ne souhaitant pas ajouter d'éléments peuvent appeler la
   * méthode \a add() sans paramètres ou cette méthode avec une vue vide.
   *
   * \param elem Les éléments à ajouter.
   */
  virtual void add(Span<const std::byte> elem) = 0;

  /*!
   * Voir \a add(Span<const std::byte> elem).
   */
  virtual void add() = 0;

  /*!
   * \brief Méthode permettant d'ajouter des éléments dans le segment d'un
   * autre sous-domaine.
   *
   * Appel collectif.
   *
   * \note Ne pas mélanger les appels de cette méthode avec les appels à
   * add().
   *
   * Deux sous-domaines ne doivent pas ajouter d'éléments dans un même
   * segment de sous-domaine.
   *
   * Si le segment ciblé est trop petit, il sera redimensionné.
   *
   * Les sous-domaines ne souhaitant pas ajouter d'éléments peuvent appeler la
   * méthode \a addToAnotherSegment() sans paramètres.
   *
   * \param rank Le sous-domaine dans lequel ajouter des éléments.
   * \param elem Les éléments à ajouter.
   */
  virtual void addToAnotherSegment(Int32 rank, Span<const std::byte> elem) = 0;

  /*!
   * Voir \a addToAnotherSegment(Int32 rank, Span<const std::byte> elem).
   */
  virtual void addToAnotherSegment() = 0;

  /*!
   * \brief Méthode permettant de réserver de l'espace mémoire dans notre
   * segment.
   *
   * Appel collectif.
   *
   * Cette méthode ne fait rien si \a new_capacity est inférieur à l'espace
   * mémoire déjà alloué pour le segment.
   * Pour les processus ne souhaitant pas augmenter l'espace mémoire
   * disponible pour leur segment, il est possible de mettre le paramètre
   * \a new_capacity à 0 ou d'utiliser la méthode \a reserve() (sans
   * arguments).
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
   * \brief Méthode permettant de redimensionner notre segment.
   *
   * Appel collectif.
   *
   * Si la taille fournie est inférieure à la taille actuelle du segment, les
   * éléments situés après la taille fournie seront supprimés.
   *
   * Pour les processus ne souhaitant pas redimensionner leur segment, il est
   * possible de mettre l'argument \a new_size à -1 ou d'appeler la méthode
   * \a resize() (sans arguments).
   *
   * \param new_size La nouvelle taille.
   */
  virtual void resize(Int64 new_size) = 0;

  /*!
   * Voir \a resize(Int64 new_size)
   */
  virtual void resize() = 0;

  /*!
   * \brief Méthode permettant de réduire l'espace mémoire réservé pour les
   * segments au minimum nécessaire.
   *
   * Appel collectif.
   */
  virtual void shrink() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
