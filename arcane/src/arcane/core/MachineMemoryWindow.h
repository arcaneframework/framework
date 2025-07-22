// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineMemoryWindow.h                                       (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire partagée entre les         */
/* processus d'un même noeud.                                                */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_MACHINEMEMORYWINDOW_H
#define ARCANE_CORE_MACHINEMEMORYWINDOW_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/Ref.h"

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
  class IMachineMemoryWindowBase;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant de créer une fenêtre mémoire partagée entre les
 * sous-domaines d'un même noeud.
 * Les segments de cette fenêtre seront contigüs en mémoire.
 *
 * \tparam Type Le type des éléments de la fenêtre.
 */
template <class Type>
class ARCANE_CORE_EXPORT MachineMemoryWindow
{
 public:

  /*!
   * \brief Constructeur.
   * \param pm Le ParallelMng contenant les processus du noeud.
   * \param nb_elem_segment Le nombre d'éléments pour le segment de ce sous-domaine.
   */
  MachineMemoryWindow(IParallelMng* pm, Int64 nb_elem_segment);

 public:

  /*!
   * \brief Méthode permettant d'obtenir une vue sur notre segment de fenêtre
   * mémoire.
   *
   * \return Une vue.
   */
  Span<Type> segmentView() const;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment de fenêtre
   * mémoire d'un autre sous-domaine du noeud.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  Span<Type> segmentView(Int32 rank) const;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur toute la fenêtre mémoire.
   *
   * \return Une vue.
   */
  Span<Type> windowView() const;

  /*!
   * \brief Méthode permettant d'obtenir une vue constante sur notre segment
   * de fenêtre mémoire.
   *
   * \return Une vue constante.
   */
  Span<const Type> segmentConstView() const;

  /*!
   * \brief Méthode permettant d'obtenir une vue constante sur le segment de
   * fenêtre mémoire d'un autre sous-domaine du noeud.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue constante.
   */
  Span<const Type> segmentConstView(Int32 rank) const;

  /*!
   * \brief Méthode permettant d'obtenir une vue constante sur toute la fenêtre
   * mémoire.
   *
   * \return Une vue constante.
   */
  Span<const Type> windowConstView() const;

  /*!
   * \brief Méthode permettant de redimensionner les segments de la fenêtre.
   * Appel collectif.
   *
   * La taille totale de la fenêtre doit être inférieure ou égale à la taille
   * d'origine.
   *
   * \param new_nb_elem La nouvelle taille de notre segment.
   */
  void resizeSegment(Integer new_nb_elem) const;

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
  Ref<MessagePassing::IMachineMemoryWindowBase> m_node_window_base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
