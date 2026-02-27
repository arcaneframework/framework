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

#ifndef ARCANE_CORE_MACHINESHMEMWINVARIABLEBASE_H
#define ARCANE_CORE_MACHINESHMEMWINVARIABLEBASE_H

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
 * la propriété "IVariable::PShMem".
 */
class ARCANE_CORE_EXPORT MachineShMemWinVariableBase
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  explicit MachineShMemWinVariableBase(IVariable* var, Int64 sizeof_type);

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

  void updateVariable(Int64 dim1);

  IVariable* variable() const;

 protected:

  IVariable* m_var = nullptr;
  Int64 m_sizeof_type = 0;
  IParallelMng* m_pm = nullptr;
  UniqueArray<Int64> m_size_var;
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
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  explicit MachineShMemWinVariable2DBase(IVariable* var, Int64 sizeof_type);

 public:

  void updateVariable(Int64 dim1, Int64 dim2);

 private:

  UniqueArray<Int64> m_dim1_var;
  UniqueArray<Int64> m_dim2_var;
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
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  explicit MachineShMemWinVariableMDBase(IVariable* var, Int64 sizeof_type);

 public:

  void updateVariable(Int64 dim1, SmallSpan<Int64, Dim> mdim);

 private:

  UniqueArray<Int64> m_dim1_var;
  FixedArray<Int64, Dim> m_mdim_var;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
