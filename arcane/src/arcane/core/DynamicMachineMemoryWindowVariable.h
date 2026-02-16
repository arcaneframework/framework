// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindowVariable.h                        (C) 2000-2026 */
/*                                                                           */
/* Classe permettant d'accéder à la partie en mémoire partagée d'une         */
/* variable.                                                                 */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOWVARIABLE_H
#define ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOWVARIABLE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/core/DynamicMachineMemoryWindowVariableBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant d'accéder à la partie en mémoire partagée d'une
 * variable.
 *
 * Il est nécessaire que cette variable soit allouée en mémoire partagée avec
 * la propriété "IVariable::PShMem".
 *
 * \tparam DataType Type de la donnée de la variable.
 */
template <class DataType>
class ARCANE_CORE_EXPORT DynamicMachineMemoryWindowVariable
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  template <class ItemType>
  explicit DynamicMachineMemoryWindowVariable(MeshVariableScalarRefT<ItemType, DataType> var)
  : m_base(var.variable())
  {}

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  template <class ItemType>
  explicit DynamicMachineMemoryWindowVariable(MeshVariableArrayRefT<ItemType, DataType> var)
  : m_base(var.variable())
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
    return m_base.machineRanks();
  }

  /*!
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  void barrier() const
  {
    m_base.barrier();
  }

  /*!
   * \brief Méthode permettant d'obtenir une vue sur notre segment.
   *
   * Équivalent à "var.asArray()".
   *
   * Appel non collectif.
   *
   * \return Une vue.
   */
  Span<DataType> segmentView() const
  {
    return asSpan<DataType>(m_base.segmentView());
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
  Span<DataType> segmentView(Int32 rank) const
  {
    return asSpan<DataType>(m_base.segmentView(rank));
  }

 private:

  DynamicMachineMemoryWindowVariableBase m_base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
