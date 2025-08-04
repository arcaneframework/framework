// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiDynamicMultiMachineMemoryWindowBaseInternal.h            (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées. Un processus peut posséder plusieurs segments.      */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIDYNAMICMULTIMACHINEMEMORYWINDOWBASEINTERNAL_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIDYNAMICMULTIMACHINEMEMORYWINDOWBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*!
 * \brief Classe basée sur MpiDynamicMachineMemoryWindowBaseInternal mais
 * pouvant gérer plusieurs segments par processus.
 *
 * Un segment est identifié par le rang de son propriétaire d'origine et par
 * un id (qui est simplement la position du segment dans la liste des segments
 * locaux).
 * Il est possible d'échanger des segments entre processus mais aussi au sein
 * d'un processus.
 * Les tableaux m_owner_segments et m_id_segments permettent de retrouver à
 * qui appartient les segments que possède chaque processus.
 *
 * Les tableaux sont tous 1D. Pour accéder à l'owner et à l'id d'un segment
 * que nous possédons, on doit calculer la position de ces informations avec
 * notre rang machine et la position de ce segment localement.
 * owner_id_pos = pos_seg + rank * m_nb_segments_per_proc
 *
 * Ensuite, m_owner_segments[owner_id_pos] est le propriétaire d'origine du
 * segment et m_id_segments[owner_id_pos] son id.
 *
 * Les informations des segments ne se déplacent pas lors des échanges de
 * segments. Alors, pour accéder à ces informations, il faut calculer leurs
 * positions :
 * infos_pos = m_id_segments[array_pos] + m_owner_segments[array_pos] * m_nb_segments_per_proc
 *
 * Ainsi, pour avoir le nombre d'éléments de ce segment, on peut y accéder
 * comme ceci :
 * m_sizeof_used_part[infos_pos]
 *
 * Cette façon de faire permet de s'assurer que tout le monde possède un
 * segment et donc que les add() ne seront pas concurrents.
 *
 * Si l'on a deux processus possèdent chacun deux segments, on aura :
 * m_id_segments    = {0, 1, 0, 1}
 * m_owner_segments = {0, 0, 1, 1}
 *
 * Si P0 veut échanger son segment ID0 avec le segment ID1 de P1, on aura
 * juste à modifier les deux tableaux :     .        .
 * m_id_segments    = {1, 1, 0, 0}
 * m_owner_segments = {1, 0, 1, 0}
 *
 * Ces modifications peuvent se faire sans atomic et l'on peut vérifier que
 * chaque échange est symétrique.
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiDynamicMultiMachineMemoryWindowBaseInternal
{

 public:

  explicit MpiDynamicMultiMachineMemoryWindowBaseInternal(SmallSpan<Int64> sizeof_segments, Int32 nb_segments_per_proc, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks);

  ~MpiDynamicMultiMachineMemoryWindowBaseInternal();

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
   * \brief Méthode permettant d'obtenir une vue sur l'un des segments que
   * nous possédons.
   *
   * Appel non collectif.
   *
   * Dans le cas d'échange de segments, il est possible d'obtenir le
   * propriétaire du segment que nous possédons avec la méthode segmentOwner(num_seg).
   *
   * \param num_seg La position (ou id) locale du segment.
   * \return Une vue.
   */
  Span<std::byte> segmentView(Int32 num_seg);

  /*!
   * \brief Méthode permettant d'obtenir une vue sur l'un des segments que
   * possède un autre processus du noeud.
   *
   * Appel non collectif.
   *
   * Dans le cas d'échange de segments, il est possible d'obtenir le
   * propriétaire du segment avec la méthode segmentOwner(rank, num_seg).
   *
   * \param rank Le rang du processus.
   * \param num_seg La position (ou id) locale du segment.
   * \return Une vue.
   */
  Span<std::byte> segmentView(Int32 rank, Int32 num_seg);

  /*!
   * \brief Méthode permettant d'obtenir une vue sur l'un des segments que
   * nous possédons.
   *
   * Appel non collectif.
   *
   * Dans le cas d'échange de segments, il est possible d'obtenir le
   * propriétaire du segment que nous possédons avec la méthode segmentOwner(num_seg).
   *
   * \param num_seg La position (ou id) locale du segment.
   * \return Une vue.
   */
  Span<const std::byte> segmentConstView(Int32 num_seg) const;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur l'un des segments que
   * possède un autre processus du noeud.
   *
   * Appel non collectif.
   *
   * Dans le cas d'échange de segments, il est possible d'obtenir le
   * propriétaire du segment avec la méthode segmentOwner(rank, num_seg).
   *
   * \param rank Le rang du processus.
   * \param num_seg La position (ou id) locale du segment.
   * \return Une vue.
   */
  Span<const std::byte> segmentConstView(Int32 rank, Int32 num_seg) const;

  /*!
   * \brief Méthode permettant d'obtenir le propriétaire d'un des segments que
   * nous possédons.
   *
   * Appel non collectif.
   *
   * Méthode utile lors de l'échange de segments.
   *
   * \param num_seg La position (ou id) locale du segment.
   * \return Le propriétaire du segment.
   */
  Int32 segmentOwner(Int32 num_seg) const;

  /*!
   * \brief Méthode permettant d'obtenir le propriétaire d'un des segments que
   * possède un autre processus du noeud.
   *
   * Appel non collectif.
   *
   * Méthode utile lors de l'échange de segments.
   *
   * \param rank Le rang du processus.
   * \param num_seg La position (ou id) locale du segment.
   * \return Le propriétaire du segment.
   */
  Int32 segmentOwner(Int32 rank, Int32 num_seg) const;

  /*!
   * \brief Méthode permettant d'obtenir l'id (ou la position d'origine sur
   * le processus d'origine) d'un des segments que nous possédons.
   *
   * Appel non collectif.
   *
   * Méthode utile lors de l'échange de segments.
   *
   * \param num_seg La position (ou id) locale du segment.
   * \return L'id d'origine du segment.
   */
  Int32 segmentPos(Int32 num_seg) const;

  /*!
   * \brief Méthode permettant d'obtenir l'id (ou la position d'origine sur
   * le processus d'origine) d'un des segments que possède un autre
   * processus du noeud.
   *
   * Appel non collectif.
   *
   * Méthode utile lors de l'échange de segments.
   *
   * \param rank Le rang du processus.
   * \param num_seg La position (ou id) locale du segment.
   * \return L'id d'origine du segment.
   */
  Int32 segmentPos(Int32 rank, Int32 num_seg) const;

  /*!
   * \brief Méthode permettant de demander l'ajout d'éléments dans l'un des
   * segments que nous possédons.
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
   * \brief Méthode permettant de demander l'échange d'un des segments que
   * nous possédons avec le segment que possède quelqu'un d'autre.
   *
   * Cet échange permet de faire des add() sur le segment d'un autre processus
   * sans avoir besoin de protéger des accès concurrents. L'échange étant
   * symétrique, on peut vérifier que chaque segment aura bien une seule
   * destination et que tout le monde possédera bien le nombre défini de
   * segments par processus.
   *
   * Pour effectuer un échange entre deux processus, le processus 0 voulant le
   * segment ID1 du processus 1 et le processus 1 voulant de segment ID2 du
   * processus 0, les processus doivent appeler cette méthode avec le rang de
   * l'autre, ainsi que les deux ids :
   * - 0 doit appeler exchangeSegmentWith(2, 1, 1) ("je veux échanger mon
   * segment ID2 avec le segment ID1 de P1),
   *
   * - 1 doit appeler exchangeSegmentWith(1, 0, 2) ("je veux échanger mon
   * segment ID1 avec le segment ID2 de P0).
   *
   * Appel non collectif et pouvant être effectué par plusieurs threads en
   * même temps (si le paramètre \a num_seg_src est différent pour chaque
   * thread).
   * Un appel à cette méthode avec un même \a num_seg_src avant l'appel à
   * \a executeExchangeSegmentWith() remplacera le premier appel.
   *
   * Un appel à executeExchangeSegmentWith() est nécessaire après le ou les
   * appels à cette méthode.
   * Il ne faut pas appeler une autre méthode \a requestX() entre temps.
   *
   * \param num_seg_src La position (ou id), locale à nous, du segment
   * que \a rank_dst souhaite.
   * \param rank_dst Le rang du processus avec qui échanger
   * \param num_seg_dst La position (ou id), locale à \a rank_dst, du segment
   * que l'on souhaite.
   */
  void requestExchangeSegmentWith(Int32 num_seg_src, Int32 rank_dst, Int32 num_seg_dst);

  /*!
   * \brief Méthode permettant d'exécuter les requêtes d'échanges.
   *
   * Appel collectif.
   */
  void executeExchangeSegmentWith();

  /*!
   * \brief Méthode permettant de réinitialiser les échanges effectués.
   *
   * Appel collectif.
   *
   * Chaque processus retrouvera ses segments à leurs positions d'origine.
   */
  void resetExchanges();

  /*!
   * \brief Méthode permettant de demander la réservation d'espace mémoire
   * pour un des segments que nous possédons.
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
   * \brief Méthode permettant de demander le redimensionnement d'un des
   * segments que nous possédons.
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

  MPI_Win m_win_owner_pos_segments;
  Span<Int32> m_owner_segments;
  Span<Int32> m_id_segments;

  MPI_Comm m_comm_machine;
  Int32 m_comm_machine_size;
  Int32 m_comm_machine_rank;

  Int32 m_sizeof_type;
  Int32 m_nb_segments_per_proc;

  ConstArrayView<Int32> m_machine_ranks;

  std::unique_ptr<Int32[]> m_exchange_requests;
  SmallSpan<Int32> m_exchange_requests_owner_segment;
  SmallSpan<Int32> m_exchange_requests_pos_segment;
  bool m_exchange_requested;

  std::unique_ptr<Span<const std::byte>[]> m_add_requests;
  SmallSpan<Span<const std::byte>> m_add_requests_span;
  bool m_add_requested;

  std::unique_ptr<Int64[]> m_resize_requests;
  SmallSpan<Int64> m_resize_requests_span;
  bool m_resize_requested;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
