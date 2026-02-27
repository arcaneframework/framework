// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariableCommon.h                                   (C) 2000-2026 */
/*                                                                           */
/* Classe permettant d'accéder à la partie en mémoire partagée d'une         */
/* variable.                                                                 */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_MACHINESHMEMWINVARIABLECOMMON_H
#define ARCANE_CORE_MACHINESHMEMWINVARIABLECOMMON_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MachineShMemWinVariableBase.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  explicit MachineShMemWinVariableCommon(IVariable* var, Int64 sizeof_type);

  virtual ~MachineShMemWinVariableCommon();

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

 protected:

  MachineShMemWinVariableBase m_base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
class ARCANE_CORE_EXPORT MachineShMemWinVariableArrayT
: public MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  explicit MachineShMemWinVariableArrayT(VariableRefArrayT<DataType> var);

 public:

  /*!
   * \brief Méthode permettant d'obtenir une vue sur notre segment.
   *
   * Équivalent à "var.asArray()".
   *
   * Appel non collectif.
   *
   * \return Une vue.
   */
  Span<DataType> segmentView() const;

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le segment d'un
   * autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  Span<DataType> segmentView(Int32 rank) const;

  void updateVariable();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
template <class ItemType, class DataType>
class ARCANE_CORE_EXPORT MachineShMemWinVariableItemT
: public MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  explicit MachineShMemWinVariableItemT(MeshVariableScalarRefT<ItemType, DataType> var);

 public:

  Span<DataType> segmentView() const;

  Span<DataType> segmentView(Int32 rank) const;

  DataType operator()(Int32 local_id);

  DataType operator()(Int32 rank, Int32 local_id);

  void updateVariable();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
class ARCANE_CORE_EXPORT MachineShMemWinVariableArray2T
{
 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  explicit MachineShMemWinVariableArray2T(VariableRefArray2T<DataType> var);

  virtual ~MachineShMemWinVariableArray2T();

 public:

  ConstArrayView<Int32> machineRanks() const;

  void barrier() const;

 public:

  Span2<DataType> segmentView() const;

  Span2<DataType> segmentView(Int32 rank) const;

  void updateVariable();

 private:

  MachineShMemWinVariable2DBase m_base;
  Int64 m_size_dim1{};
  Int64 m_size_dim2{};
  VariableRefArray2T<DataType> m_vart;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
template <class ItemType, class DataType>
class ARCANE_CORE_EXPORT MachineShMemWinVariableItemArrayT
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PShMem".
   */
  explicit MachineShMemWinVariableItemArrayT(MeshVariableArrayRefT<ItemType, DataType> var);

  virtual ~MachineShMemWinVariableItemArrayT();

 public:

  ConstArrayView<Int32> machineRanks() const;

  void barrier() const;

 public:

  Span2<DataType> segmentView() const;

  Span2<DataType> segmentView(Int32 rank) const;
  Span<DataType> segmentView1D() const;

  Span<DataType> segmentView1D(Int32 rank) const;

  Span<DataType> operator()(Int32 local_id);

  Span<DataType> operator()(Int32 rank, Int32 local_id);

  void updateVariable();

 private:

  MachineShMemWinVariableMDBase<1> m_base;
  Int64 m_size_dim1{};
  Int64 m_size_dim2{};
  MeshVariableArrayRefT<ItemType, DataType> m_vart;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
