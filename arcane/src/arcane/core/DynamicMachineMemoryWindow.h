﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindow.h                                (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées.                                                     */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOW_H
#define ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOW_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/DynamicMachineMemoryWindowBase.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/Ref.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
class DynamicMachineMemoryWindow
{

 public:

  /*!
   * \brief Constructeur.
   * \param pm Le parallelMng à utiliser.
   * \param nb_elem_segment Le nombre d'éléments initial.
   */
  DynamicMachineMemoryWindow(IParallelMng* pm, Int64 nb_elem_segment)
  : m_impl(pm, nb_elem_segment, static_cast<Int32>(sizeof(Type)))
  {}

  /*!
   * \brief Constructeur.
   * \param pm Le parallelMng à utiliser.
   */
  explicit DynamicMachineMemoryWindow(IParallelMng* pm)
  : m_impl(pm, 0, static_cast<Int32>(sizeof(Type)))
  {}

 public:

  /*!
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * Appel non collectif.
   *
   * \return Une vue contenant les ids des rangs.
   */
  ConstArrayView<Int32> machineRanks() const
  {
    return m_impl.machineRanks();
  }

  /*!
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  void barrier() const
  {
    m_impl.barrier();
  }
  /*!
   * \brief Méthode permettant d'obtenir une vue sur notre segment.
   *
   * Appel non collectif.
   *
   * \return Une vue.
   */
  Span<Type> segmentView()
  {
    return asSpan<Type>(m_impl.segmentView());
  }

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment d'un
   * autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  Span<Type> segmentView(Int32 rank)
  {
    return asSpan<Type>(m_impl.segmentView(rank));
  }

  /*!
   * \brief Méthode permettant d'obtenir une vue sur notre segment.
   *
   * Appel non collectif.
   *
   * \return Une vue.
   */
  Span<const Type> segmentConstView() const
  {
    return asSpan<const Type>(m_impl.segmentConstView());
  }

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment d'un
   * autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  Span<const Type> segmentConstView(Int32 rank) const
  {
    return asSpan<const Type>(m_impl.segmentConstView(rank));
  }

  /*!
   * \brief Méthode permettant d'ajouter des élements dans notre segment.
   *
   * Appel collectif.
   *
   * \note Les méthodes add(..) et addToAnotherSegment(..) ne se mélangent pas.
   *
   * Si le segment est trop petit, il sera redimensionné.
   *
   * Les sous-domaines ne souhaitant pas ajouter d'éléments peuvent appeler la
   * méthode \a add() sans paramètres ou cette méthode avec une vue vide.
   *
   * \param elem Les éléments à ajouter.
   */
  void add(Span<const Type> elem)
  {
    const Span<const std::byte> span_bytes(reinterpret_cast<const std::byte*>(elem.data()), elem.sizeBytes());
    m_impl.add(span_bytes);
  }

  /*!
   * \brief Méthode à appeler par le ou les sous-domaines ne souhaitant pas ajouter
   * d'éléments dans son segment.
   *
   * Appel collectif.
   *
   * \note Les méthodes add(..) et addToAnotherSegment(..) ne se mélangent pas.
   *
   * Voir la documentation de \a add(Span<const std::byte> elem).
   */
  void add()
  {
    m_impl.add();
  }

  /*!
   * \brief Méthode permettant d'ajouter des éléments dans le segment d'un
   * autre sous-domaine.
   *
   * Appel collectif.
   *
   * \note Les méthodes add(..) et addToAnotherSegment(..) ne se mélangent pas.
   *
   * Deux sous-domaines ne doivent pas ajouter d'éléments dans un même
   * segment de sous-domaine.
   *
   * Si le segment ciblé est trop petit, il sera redimensionné.
   *
   * Les sous-domaines ne souhaitant pas ajouter d'éléments peuvent appeler la
   * méthode \a addToAnotherSegment() sans paramètres.
   *
   * \param rank Le rang du sous-domaine avec le segment à modifier.
   * \param elem Les éléments à ajouter.
   */
  void addToAnotherSegment(Int32 rank, Span<const Type> elem)
  {
    const Span<const std::byte> span_bytes(reinterpret_cast<const std::byte*>(elem.data()), elem.sizeBytes());
    m_impl.addToAnotherSegment(rank, span_bytes);
  }

  /*!
   * \brief Méthode à appeler par le ou les sous-domaines ne souhaitant pas ajouter
   * d'éléments dans le segment d'un autre sous-domaine.
   *
   * Appel collectif.
   *
   * \note Les méthodes add(..) et addToAnotherSegment(..) ne se mélangent pas.
   *
   * Voir la documentation de \a addToAnotherSegment(Int32 rank, Span<const Type> elem).
   */
  void addToAnotherSegment()
  {
    m_impl.addToAnotherSegment();
  }

  /*!
   * \brief Méthode permettant de réserver de l'espace mémoire dans notre segment.
   *
   * Appel collectif.
   *
   * Cette méthode ne fait rien si \a new_capacity est inférieur à l'espace
   * mémoire déjà alloué pour le segment.
   * Pour les sous-domaines ne souhaitant pas augmenter l'espace mémoire
   * disponible pour leur segment, il est possible de mettre le paramètre
   * \a new_capacity à 0 ou d'utiliser la méthode \a reserve() (sans
   * arguments).
   *
   * L'espace qui sera réservé aura une taille supérieur ou égale à
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
  void reserve(Int64 new_capacity)
  {
    m_impl.reserve(new_capacity);
  }

  /*!
   * \brief Méthode à appeler par le ou les sous-domaines ne souhaitant pas réserver
   * davantage de mémoire pour leurs segments.
   *
   * Appel collectif.
   *
   * Voir la documentation de \a reserve(Int64 new_capacity).
   */
  void reserve()
  {
    m_impl.reserve();
  }

  /*!
   * \brief Méthode permettant de redimensionner notre segment.
   *
   * Appel collectif.
   *
   * Si la taille fournie est inférieure à la taille actuelle du segment, les
   * éléments situés après la taille fournie seront supprimés.
   *
   * Pour les sous-domaines ne souhaitant pas redimensionner leur segment, il est
   * possible de mettre l'argument \a new_size à -1 ou d'appeler la méthode
   * \a resize() (sans arguments).
   *
   * \param new_nb_elem La nouvelle taille.
   */
  void resize(Int64 new_nb_elem)
  {
    m_impl.resize(new_nb_elem);
  }

  /*!
   * \brief Méthode à appeler par le ou les sous-domaines ne souhaitant pas
   * redimensionner leurs segments.
   *
   * Appel collectif.
   *
   * Voir la documentation de \a resize(Int64 new_nb_elem).
   */
  void resize()
  {
    m_impl.resize();
  }

  /*!
   * \brief Méthode permettant de réduire l'espace mémoire réservé pour les
   * segments au minimum nécessaire.
   *
   * Appel collectif.
   */
  void shrink()
  {
    m_impl.shrink();
  }

 private:

  DynamicMachineMemoryWindowBase m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
