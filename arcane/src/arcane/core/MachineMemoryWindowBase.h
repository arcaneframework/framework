// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineMemoryWindowBase.h                                   (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire partagée entre les         */
/* processus d'un même noeud.                                                */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_MACHINEMEMORYWINDOWBASE_H
#define ARCANE_CORE_MACHINEMEMORYWINDOWBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/Ref.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;
class IParallelMngInternal;
namespace MessagePassing
{
  class IMachineMemoryWindowBaseInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant de créer une fenêtre mémoire partagée entre les
 * sous-domaines d'un même noeud.
 * Les segments de cette fenêtre seront contigüs en mémoire.
 */
class ARCANE_CORE_EXPORT MachineMemoryWindowBase
{

 public:

  /*!
   * \brief Constructeur.
   * \param pm Le ParallelMng contenant les processus du noeud.
   * \param sizeof_segment La taille du segment de ce sous-domaine (en octet).
   * \param sizeof_elem La taille d'un élément (en octet).
   */
  MachineMemoryWindowBase(IParallelMng* pm, Int64 sizeof_segment, Int32 sizeof_elem);

 public:

  /*!
   * \brief Méthode permettant d'obtenir une vue sur notre segment de fenêtre
   * mémoire.
   *
   * \return Une vue.
   */
  Span<std::byte> segmentView();

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment de fenêtre
   * mémoire d'un autre sous-domaine du noeud.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  Span<std::byte> segmentView(Int32 rank);

  /*!
   * \brief Méthode permettant d'obtenir une vue sur toute la fenêtre mémoire.
   *
   * \return Une vue.
   */
  Span<std::byte> windowView();

  /*!
   * \brief Méthode permettant d'obtenir une vue constante sur notre segment
   * de fenêtre mémoire.
   *
   * \return Une vue constante.
   */
  Span<const std::byte> segmentConstView() const;

  /*!
   * \brief Méthode permettant d'obtenir une vue constante sur le segment de
   * fenêtre mémoire d'un autre sous-domaine du noeud.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue constante.
   */
  Span<const std::byte> segmentConstView(Int32 rank) const;

  /*!
   * \brief Méthode permettant d'obtenir une vue constante sur toute la fenêtre
   * mémoire.
   *
   * \return Une vue constante.
   */
  Span<const std::byte> windowConstView() const;

  /*!
   * \brief Méthode permettant de redimensionner les segments de la fenêtre.
   * Appel collectif.
   *
   * La taille totale de la fenêtre doit être inférieure ou égale à la taille
   * d'origine.
   *
   * \param new_size La nouvelle taille de notre segment (en octet).
   */
  void resizeSegment(Integer new_size);

  /*!
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * L'ordre des processus de la vue retournée correspond à l'ordre des
   * segments dans la fenêtre.
   *
   * \return Une vue contenant les ids des rangs.
   */
  ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  void barrier() const;

 private:

  IParallelMngInternal* m_pm_internal;
  Ref<MessagePassing::IMachineMemoryWindowBaseInternal> m_node_window_base;
  Int32 m_sizeof_elem;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
