// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariableBase.h                               (C) 2000-2026 */
/*                                                                           */
/* Allocateur mémoire utilisant la classe MachineShMemWinBase.               */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_INTERNAL_MACHINESHMEMWINVARIABLEBASE_H
#define ARCANE_CORE_INTERNAL_MACHINESHMEMWINVARIABLEBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/FixedArray.h"

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
 * la propriété "IVariable::PInShMem".
 */
class ARCANE_CORE_EXPORT MachineShMemWinVariableBase
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariableBase(IVariable* var);

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
   * \brief Méthode permettant d'obtenir une vue sur le segment d'un
   * autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  Span<std::byte> segmentView(Int32 rank) const;

  /*!
   * \brief
   * \param nb_elem_dim1 En nb éléments
   */
  void updateVariable(Int64 nb_elem_dim1, Int64 sizeof_elem);

  IVariable* variable() const;

 protected:

  IVariable* m_var = nullptr;
  IParallelMng* m_pm = nullptr;

  // En octet.
  UniqueArray<Int64> m_sizeof_var;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT MachineShMemWinVariable2DBase
: public MachineShMemWinVariableBase
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariable2DBase(IVariable* var);

 public:

  /*!
   * \brief
   * \param nb_elem_dim1 En nb elements
   * \param nb_elem_dim2 En nb elements
   */
  void updateVariable(Int64 nb_elem_dim1, Int64 nb_elem_dim2, Int64 sizeof_elem);

  ArrayView<Int64> nbElemDim1();
  ArrayView<Int64> nbElemDim2();

 private:

  // En nb éléments.
  UniqueArray<Int64> m_nb_elem_dim1;
  UniqueArray<Int64> m_nb_elem_dim2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <Int32 Dim>
class ARCANE_CORE_EXPORT MachineShMemWinVariableMDBase
: public MachineShMemWinVariableBase
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariableMDBase(IVariable* var);

 public:

  /*!
   * \brief
   * \param nb_elem_dim1 En nb elements
   * \param nb_elem_mdim En nb elements
   */
  void updateVariable(Int64 nb_elem_dim1, SmallSpan<Int64, Dim> nb_elem_mdim, Int64 sizeof_elem);

  ArrayView<Int64> nbElemDim1();

 private:

  // En nb éléments.
  UniqueArray<Int64> m_nb_elem_dim1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
