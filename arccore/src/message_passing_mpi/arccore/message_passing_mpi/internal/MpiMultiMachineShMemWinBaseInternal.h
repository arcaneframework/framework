// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMultiMachineShMemWinBaseInternal.h            (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées. Un processus peut posséder plusieurs segments.      */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMULTIMACHINESHMEMWINBASEINTERNAL_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMULTIMACHINESHMEMWINBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*!
 * \brief Classe basée sur MpiMachineShMemWinBaseInternal mais
 * pouvant gérer plusieurs segments par processus.
 *
 * Un segment est identifié par le rang de son propriétaire d'origine et par
 * un id (qui est simplement la position du segment dans la liste des segments
 * locaux).
 *
 * Les tableaux sont tous 1D. Pour accéder aux infos d'un de nos segments,
 * on doit calculer la position de ces informations avec
 * notre rang machine et la position de ce segment localement.
 * infos_pos = pos_seg + rank * m_nb_segments_per_proc
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiMultiMachineShMemWinBaseInternal
{

 public:

  explicit MpiMultiMachineShMemWinBaseInternal(SmallSpan<Int64> sizeof_segments, Int32 nb_segments_per_proc, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks);

  ~MpiMultiMachineShMemWinBaseInternal();

 public:

  /*!
   * \brief Méthode permettant d'obtenir la taille d'un élement de la fenêtre.
   *
   * Appel non collectif.
   *
   * \return La taille d'un élement.
   */
  Int32 sizeofOneElem() const;

  /*!
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * Appel non collectif.
   *
   * \return Une vue contenant les ids des rangs.
   */
  ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Méthode permettant d'attendre que tous les processus
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  void barrier() const;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur l'un de nos segments.
   *
   * Appel non collectif.
   *
   * \param num_seg La position (ou id) du segment.
   * \return Une vue.
   */
  Span<std::byte> segmentView(Int32 num_seg);

  /*!
   * \brief Méthode permettant d'obtenir une vue sur l'un des segments
   * d'un autre processus du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du processus.
   * \param num_seg La position (ou id) locale du segment.
   * \return Une vue.
   */
  Span<std::byte> segmentView(Int32 rank, Int32 num_seg);

  /*!
   * \brief Méthode permettant d'obtenir une vue sur l'un de nos segments.
   *
   * Appel non collectif.
   *
   * \param num_seg La position (ou id) locale du segment.
   * \return Une vue.
   */
  Span<const std::byte> segmentConstView(Int32 num_seg) const;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur l'un des segments d'un
   * autre processus du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du processus.
   * \param num_seg La position (ou id) locale du segment.
   * \return Une vue.
   */
  Span<const std::byte> segmentConstView(Int32 rank, Int32 num_seg) const;

  /*!
   * \brief Méthode permettant de demander l'ajout d'éléments dans l'un de nos
   * segments.
   *
   * Appel non collectif et pouvant être effectué par plusieurs threads en
   * même temps (si le paramètre \a num_seg est différent pour chaque thread).
   * Un appel à cette méthode avec un même \a num_seg avant l'appel à
   * \a executeAdd() remplacera le premier appel.
   *
   * Un appel à executeAdd() est nécessaire après le ou les appels à cette
   * méthode.
   * Il ne faut pas appeler une autre méthode \a requestX() entre temps.
   *
   * \param num_seg La position (ou id) locale du segment.
   * \param elem Les éléments à ajouter.
   */
  void requestAdd(Int32 num_seg, Span<const std::byte> elem);

  /*!
   * \brief Méthode permettant d'exécuter les requêtes d'ajout.
   *
   * Appel collectif.
   */
  void executeAdd();

  /*!
   * \brief Méthode permettant de demander l'ajout d'éléments dans un des
   * segments de la fenêtre.
   *
   * Appel non collectif et pouvant être effectué par plusieurs threads en
   * même temps (si le paramètre \a thread est différent pour chaque thread).
   * Un appel à cette méthode avec un même \a thread avant l'appel à
   * \a executeaddToAnotherSegment() remplacera le premier appel.
   *
   * Un appel à executeaddToAnotherSegment() est nécessaire après le ou les appels à
   * cette méthode.
   * Il ne faut pas appeler une autre méthode \a requestX() entre temps.
   *
   * Deux sous-domaines ne doivent pas ajouter d'éléments dans un même
   * segment de sous-domaine.
   *
   * \param thread Le thread qui demande l'ajout. TODO Trouver un autre moyen que ce paramètre.
   * \param rank Le rang du processus propriétaire du segment à modifier.
   * \param num_seg La position (ou id) locale du segment à modifier.
   * \param elem Les éléments à ajouter.
   */
  void requestAddToAnotherSegment(Int32 thread, Int32 rank, Int32 num_seg, Span<const std::byte> elem);

  /*!
   * \brief Méthode permettant d'exécuter les requêtes d'ajout dans les
   * segments d'autres processus.
   *
   * Appel collectif.
   */
  void executeAddToAnotherSegment();

  /*!
   * \brief Méthode permettant de demander la réservation d'espace mémoire
   * pour un de nos segments.
   *
   * Cette méthode ne fait rien si \a new_capacity est inférieur à l'espace
   * mémoire déjà alloué pour le segment.
   *
   * MPI réservera un espace avec une taille supérieur ou égale à
   * \a new_capacity.
   *
   * Cette méthode ne redimensionne pas le segment, il faudra toujours passer
   * par les méthodes add() pour ajouter des éléments.
   *
   * Pour redimensionner le segment, les méthodes \a resize() sont
   * disponibles.
   *
   * Appel non collectif et pouvant être effectué par plusieurs threads en
   * même temps (si le paramètre \a num_seg est différent pour chaque thread).
   * Un appel à cette méthode avec un même \a num_seg avant l'appel à
   * \a executeReserve() remplacera le premier appel.
   *
   * Un appel à executeReserve() est nécessaire après le ou les appels à cette
   * méthode.
   * Il ne faut pas appeler une autre méthode \a requestX() entre temps.
   *
   * \param num_seg La position (ou id) locale du segment.
   * \param new_capacity La nouvelle capacité demandée.
   */
  void requestReserve(Int32 num_seg, Int64 new_capacity);

  /*!
   * \brief Méthode permettant d'exécuter les requêtes de réservation.
   *
   * Appel collectif.
   */
  void executeReserve();

  /*!
   * \brief Méthode permettant de demander le redimensionnement d'un de nos
   * segments.
   *
   * Si la taille fournie est inférieure à la taille actuelle du segment, les
   * éléments situés après la taille fournie seront supprimés.
   *
   * Si la taille fournie est supérieur à l'espace mémoire réservé au segment,
   * un realloc sera effectué.
   *
   * Appel non collectif et pouvant être effectué par plusieurs threads en
   * même temps (si le paramètre \a num_seg est différent pour chaque thread).
   * Un appel à cette méthode avec un même \a num_seg avant l'appel à
   * \a executeResize() remplacera le premier appel.
   *
   * Un appel à executeResize() est nécessaire après le ou les appels à cette
   * méthode.
   * Il ne faut pas appeler une autre méthode \a requestX() entre temps.
   *
   * \param num_seg La position (ou id) locale du segment.
   * \param new_size La nouvelle taille.
   */
  void requestResize(Int32 num_seg, Int64 new_size);

  /*!
   * \brief Méthode permettant d'exécuter les requêtes de redimensionnement.
   *
   * Appel collectif.
   */
  void executeResize();

  /*!
   * \brief Méthode permettant de réduire l'espace mémoire réservé pour les
   * segments au minimum nécessaire.
   *
   * Appel collectif.
   */
  void executeShrink();

 private:

  void _requestRealloc(Int32 owner_pos_segment, Int64 new_capacity) const;
  void _requestRealloc(Int32 owner_pos_segment) const;
  void _executeRealloc();
  void _realloc();

  Int32 _worldToMachine(Int32 world) const;
  Int32 _machineToWorld(Int32 machine) const;

 private:

  UniqueArray<MPI_Win> m_all_mpi_win;
  // Tableau avec les vues sur les segments. La taille des vues correspond à tout
  // l'espace mémoire réservé.
  UniqueArray<Span<std::byte>> m_reserved_part_span;

  MPI_Win m_win_need_resize;
  Span<Int64> m_need_resize;

  MPI_Win m_win_actual_sizeof;
  Span<Int64> m_sizeof_used_part;

  MPI_Win m_win_target_segments;
  Span<Int32> m_target_segments;

  MPI_Comm m_comm_machine;
  Int32 m_comm_machine_size = 0;
  Int32 m_comm_machine_rank = 0;

  Int32 m_sizeof_type = 0;
  Int32 m_nb_segments_per_proc = 0;

  ConstArrayView<Int32> m_machine_ranks;

  UniqueArray<Span<const std::byte>> m_add_requests;
  bool m_add_requested = false;

  UniqueArray<Int64> m_resize_requests;
  bool m_resize_requested = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
