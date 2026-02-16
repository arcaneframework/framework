// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindowVariableBase.h                 (C) 2000-2026 */
/*                                                                           */
/* Allocateur mémoire utilisant la classe DynamicMachineMemoryWindowBase.    */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOWVARIABLEBASE_H
#define ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOWVARIABLEBASE_H

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
 * \brief Classe permettant d'accéder à la partie en mémoire partagée entre
 * sous-domaines d'un même noeud d'une variable.
 *
 * Il est nécessaire que cette variable soit allouée en mémoire partagée avec
 * la propriété "IVariable::PShMem".
 */
class ARCANE_CORE_EXPORT DynamicMachineMemoryWindowVariableBase
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  explicit DynamicMachineMemoryWindowVariableBase(IVariable* var);

 public:

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
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  void barrier() const;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur notre segment.
   *
   * Appel non collectif.
   *
   * \return Une vue.
   */
  Span<std::byte> segmentView() const;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment d'un
   * autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  Span<std::byte> segmentView(Int32 rank) const;

 private:

  IVariable* m_var;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
