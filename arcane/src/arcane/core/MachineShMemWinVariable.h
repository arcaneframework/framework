// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariable.h                                   (C) 2000-2026 */
/*                                                                           */
/* Classe permettant d'accéder à la partie en mémoire partagée d'une         */
/* variable.                                                                 */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_MACHINESHMEMWINVARIABLE_H
#define ARCANE_CORE_MACHINESHMEMWINVARIABLE_H

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

/*!
 * \brief Classe permettant d'accéder aux éléments partagés de la variable
 * en mémoire partagée.
 *
 * Pour avoir accès à toutes les propriétés, il est conseillé d'utiliser une
 * des classes enfants :
 * - \a MachineShMemWinVariableArrayT pour les variables tableaux sans
 *   support,
 * - \a MachineShMemWinVariableItemT pour les variables au maillage.
 */
class ARCANE_CORE_EXPORT MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
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
 * \brief Classe permettant d'accéder aux éléments partagés de la variable
 * en mémoire partagée.
 *
 * Il est nécessaire que cette variable soit allouée en mémoire partagée avec
 * la propriété "IVariable::PInShMem".
 *
 * Cette classe fonctionne pour les variables tableaux sans support.
 *
 * Si le maillage change lorsqu'un objet de ce type est utilisé, il est
 * nécessaire d'appeler la méthode \a updateVariable().
 */
template <class DataType>
class ARCANE_CORE_EXPORT MachineShMemWinVariableArrayT
: public MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "PInShMem".
   */
  explicit MachineShMemWinVariableArrayT(VariableRefArrayT<DataType> var);

 public:

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le tableau d'un
   * autre sous-domaine du noeud.
   *
   * Équivalent à "var.asArray()" mais d'un autre sous-domaine.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  Span<DataType> view(Int32 rank) const;

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un changement
   * dans le maillage.
   *
   * Appel collectif.
   */
  void updateVariable();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant d'accéder aux éléments partagés de la variable
 * en mémoire partagée.
 *
 * Il est nécessaire que cette variable soit allouée en mémoire partagée avec
 * la propriété "IVariable::PInShMem".
 *
 * Cette classe fonctionne pour les variables scalaire au maillage.
 *
 * Si le maillage change lorsqu'un objet de ce type est utilisé, il est
 * nécessaire d'appeler la méthode \a updateVariable().
 */
template <class ItemType, class DataType>
class ARCANE_CORE_EXPORT MachineShMemWinVariableItemT
: public MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariableItemT(MeshVariableScalarRefT<ItemType, DataType> var);

 public:

  /*!
   * \brief Méthode permettant d'obtenir une vue sur la variable d'un
   * autre sous-domaine du noeud.
   *
   * Équivalent à "var.asArray()" mais d'un autre sous-domaine.
   *
   * \warning Attention : pour accéder aux éléments de la vue, il est
   *          nécessaire d'utiliser les local_ids de l'autre sous-domaine !
   *          Ne pas utiliser les local_ids de notre sous-domaine !
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue.
   */
  Span<DataType> view(Int32 rank) const;

  /*!
   * \brief Méthode permettant d'obtenir un élément de la variable d'un autre
   * sous-domaine.
   *
   * \warning Attention : le local_id correspond au local_id du sous-domaine
   *          \a rank ! Ne surtout pas utiliser un local_id de notre
   *          sous-domaine pour accéder aux éléments de la vue !
   *
   * \note Si plusieurs itérations sont nécessaires pour un même rang, il est
   *       préférable de récupérer une vue via \a segmentView(Int32 rank).
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine de la variable ciblée.
   * \param notlocal_id Le local_id du sous-domaine \a rank.
   * \return L'élément de l'item.
   */
  DataType operator()(Int32 rank, Int32 notlocal_id);

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un changement
   * dans le maillage.
   *
   * Appel collectif.
   */
  void updateVariable();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant d'accéder aux éléments partagés de la variable
 * en mémoire partagée.
 *
 * Il est nécessaire que cette variable soit allouée en mémoire partagée avec
 * la propriété "IVariable::PInShMem".
 *
 * Cette classe fonctionne pour les variables tableaux 2D sans support.
 *
 * Si le maillage change lorsqu'un objet de ce type est utilisé, il est
 * nécessaire d'appeler la méthode \a updateVariable().
 */
template <class DataType>
class ARCANE_CORE_EXPORT MachineShMemWinVariableArray2T
{
 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariableArray2T(VariableRefArray2T<DataType> var);

  virtual ~MachineShMemWinVariableArray2T();

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

 public:

  /*!
   * \brief Méthode permettant d'obtenir une vue sur le tableau d'un
   * autre sous-domaine du noeud.
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue 2D.
   */
  Span2<DataType> view(Int32 rank) const;

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un changement
   * dans le maillage.
   *
   * Appel collectif.
   */
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
 * \brief Classe permettant d'accéder aux éléments partagés de la variable
 * en mémoire partagée.
 *
 * Il est nécessaire que cette variable soit allouée en mémoire partagée avec
 * la propriété "IVariable::PInShMem".
 *
 * Cette classe fonctionne pour les variables tableaux au maillage.
 *
 * Si le maillage change lorsqu'un objet de ce type est utilisé, il est
 * nécessaire d'appeler la méthode \a updateVariable().
 */
template <class ItemType, class DataType>
class ARCANE_CORE_EXPORT MachineShMemWinVariableItemArrayT
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariableItemArrayT(MeshVariableArrayRefT<ItemType, DataType> var);

  virtual ~MachineShMemWinVariableItemArrayT();

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

 public:

  /*!
   * \brief Méthode permettant d'obtenir une vue sur la variable d'un
   * autre sous-domaine du noeud.
   *
   * Équivalent à "var.asArray()" mais d'un autre sous-domaine.
   * Le premier indice correspond au local_id, le second indice est la
   * position de l'élément dans le tableau de l'item.
   *
   * \warning Attention : pour accéder aux éléments de la vue, il est
   *          nécessaire d'utiliser les local_ids de l'autre sous-domaine !
   *          Ne pas utiliser les local_ids de notre sous-domaine !
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine.
   * \return Une vue 2D.
   */
  Span2<DataType> view(Int32 rank) const;

  /*!
   * \brief Méthode permettant d'obtenir le tableau d'un item d'un autre
   * sous-domaine.
   *
   * \warning Attention : le local_id correspond au local_id du sous-domaine
   *          \a rank ! Ne surtout pas utiliser un local_id de notre
   *          sous-domaine pour accéder aux éléments de la vue !
   *
   * \note Si plusieurs itérations sont nécessaires pour un même rang, il est
   *       préférable de récupérer une vue via \a segmentView(Int32 rank).
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine de la variable ciblée.
   * \param notlocal_id Le local_id du sous-domaine \a rank.
   * \return Le tableau de l'item.
   */
  Span<DataType> operator()(Int32 rank, Int32 notlocal_id);

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
