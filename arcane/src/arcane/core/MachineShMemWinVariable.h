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

#include "arcane/core/ArcaneTypes.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/NumArray.h"

#include "arcane/core/MeshMDVariableRef.h"

#include "arccore/base/FixedArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MachineShMemWinVariableBase;
class MachineShMemWinVariable2DBase;
class MachineShMemWinVariableMDBase;

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
  explicit MachineShMemWinVariableCommon(IVariable* var);

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

  Ref<MachineShMemWinVariableBase> m_base;
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
  ~MachineShMemWinVariableArrayT() override;

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

 private:

  VariableRefArrayT<DataType> m_vart;
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
class ARCANE_CORE_EXPORT MachineShMemWinMeshVariableScalarT
: public MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinMeshVariableScalarT(MeshVariableScalarRefT<ItemType, DataType> var);

  ~MachineShMemWinMeshVariableScalarT() override;

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

 private:

  MeshVariableScalarRefT<ItemType, DataType> m_vart;
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

  ~MachineShMemWinVariableArray2T();

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

  Ref<MachineShMemWinVariable2DBase> m_base;
  VariableRefArray2T<DataType> m_vart;
  ConstArrayView<Int64> m_nb_elem_dim1;
  ConstArrayView<Int64> m_nb_elem_dim2;
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
class ARCANE_CORE_EXPORT MachineShMemWinMeshVariableArrayT
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinMeshVariableArrayT(MeshVariableArrayRefT<ItemType, DataType> var);

  ~MachineShMemWinMeshVariableArrayT();

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

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un changement
   * dans le maillage.
   *
   * Appel collectif.
   */
  void updateVariable();

 private:

  Ref<MachineShMemWinVariableMDBase> m_base;
  MeshVariableArrayRefT<ItemType, DataType> m_vart;
  ConstArrayView<Int64> m_nb_elem_dim1;
  Int32 m_nb_elem_dim2{};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType, class Extents>
class ARCANE_CORE_EXPORT MachineShMemWinMDVariableT
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinMDVariableT(IVariable* var);

  virtual ~MachineShMemWinMDVariableT();

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
   * Le premier indice correspond au local_id, les autres indices sont la
   * position de l'élément dans le tableau de l'item.
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
  // template <class X = MDDimType<Extents::rank() + 1>::DimType>
  // MDSpan<DataType, X> view(Int32 rank) const;

  MDSpan<DataType, typename MDDimType<Extents::rank() + 1>::DimType> view(Int32 rank) const;

  /*!
   * \brief Méthode permettant d'obtenir le tableau multi-dimensionnel d'un
   * item d'un autre sous-domaine.
   *
   * \warning Attention : le local_id correspond au local_id du sous-domaine
   *          \a rank ! Ne surtout pas utiliser un local_id de notre
   *          sous-domaine pour accéder aux éléments de la vue !
   *
   * \note Si plusieurs itérations sont nécessaires pour un même rang, il est
   *       préférable de récupérer une vue via \a view(Int32 rank).
   *
   * Appel non collectif.
   *
   * \param rank Le rang du sous-domaine de la variable ciblée.
   * \param notlocal_id Le local_id du sous-domaine \a rank.
   * \return Le tableau MD de l'item.
   */
  virtual MDSpan<DataType, Extents> operator()(Int32 rank, Int32 notlocal_id);

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un changement
   * dans le maillage.
   *
   * Appel collectif.
   */
  void updateVariable(Int64 nb_elem_dim1, Int32 nb_elem_dim2, SmallSpan<const Int32> shape_dim2);

 private:

  Ref<MachineShMemWinVariableMDBase> m_base;
  ConstArrayView<Int64> m_nb_elem_dim1;
  Int32 m_nb_elem_dim2{};
  std::array<Int32, Extents::rank()> m_shape_dim2{};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType, class Extents>
class ARCANE_CORE_EXPORT MachineShMemWinMeshMDVariableT
: public MachineShMemWinMDVariableT<ItemType, DataType, Extents>
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinMeshMDVariableT(MeshMDVariableRefT<ItemType, DataType, Extents> var);

  ~MachineShMemWinMeshMDVariableT() override;

 public:

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un changement
   * dans le maillage.
   *
   * Appel collectif.
   */
  void updateVariable();

 private:

  MeshMDVariableRefT<ItemType, DataType, Extents> m_vart;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
