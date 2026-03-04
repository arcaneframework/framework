// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariable.h                                   (C) 2000-2026 */
/*                                                                           */
/* Classes permettant d'exploiter l'objet MachineShMemWinVariable pointé de  */
/* la zone mémoire des variables en mémoire partagée.                        */
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
 * Pour avoir accès à toutes les propriétés, il est nécessaire d'utiliser une
 * des classes enfants :
 * - \a MachineShMemWinVariableArrayT pour les variables tableaux sans
 *   support,
 * - \a MachineShMemWinVariableItemT pour les variables au maillage.
 */
class ARCANE_CORE_EXPORT MachineShMemWinVariableCommon
{

 protected:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinVariableCommon(IVariable* var);

 public:

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
 * Si la taille de la variable change lorsqu'un objet de ce type est utilisé,
 * il est nécessaire d'appeler la méthode \a updateVariable().
 */
template <class DataType>
class MachineShMemWinVariableArrayT
: public MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinVariableArrayT(VariableRefArrayT<DataType> var);
  ARCANE_CORE_EXPORT ~MachineShMemWinVariableArrayT() override;

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
  ARCANE_CORE_EXPORT Span<DataType> view(Int32 rank) const;

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un
   * redimensionnement de la variable.
   *
   * Appel collectif.
   */
  ARCANE_CORE_EXPORT void updateVariable();

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
class MachineShMemWinMeshVariableScalarT
: public MachineShMemWinVariableCommon
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinMeshVariableScalarT(MeshVariableScalarRefT<ItemType, DataType> var);

  ARCANE_CORE_EXPORT ~MachineShMemWinMeshVariableScalarT() override;

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
  ARCANE_CORE_EXPORT Span<DataType> view(Int32 rank) const;

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
  ARCANE_CORE_EXPORT DataType operator()(Int32 rank, Int32 notlocal_id);

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un changement
   * dans le maillage.
   *
   * Appel collectif.
   */
  ARCANE_CORE_EXPORT void updateVariable();

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
 * Si la taille de la variable change lorsqu'un objet de ce type est utilisé,
 * il est nécessaire d'appeler la méthode \a updateVariable().
 */
template <class DataType>
class MachineShMemWinVariableArray2T
{
 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinVariableArray2T(VariableRefArray2T<DataType> var);

  ARCANE_CORE_EXPORT ~MachineShMemWinVariableArray2T();

 public:

  /*!
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * Appel non collectif.
   *
   * \return Une vue contenant les ids des rangs.
   */
  ARCANE_CORE_EXPORT ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  ARCANE_CORE_EXPORT void barrier() const;

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
  ARCANE_CORE_EXPORT Span2<DataType> view(Int32 rank) const;

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un
   * redimensionnement de la variable.
   *
   * Appel collectif.
   */
  ARCANE_CORE_EXPORT void updateVariable();

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
 * Si le maillage et/ou la taille de la variable change lorsqu'un objet de ce
 * type est utilisé, il est nécessaire d'appeler la méthode \a updateVariable().
 */
template <class ItemType, class DataType>
class MachineShMemWinMeshVariableArrayT
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinMeshVariableArrayT(MeshVariableArrayRefT<ItemType, DataType> var);

  ARCANE_CORE_EXPORT ~MachineShMemWinMeshVariableArrayT();

 public:

  /*!
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * Appel non collectif.
   *
   * \return Une vue contenant les ids des rangs.
   */
  ARCANE_CORE_EXPORT ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  ARCANE_CORE_EXPORT void barrier() const;

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
  ARCANE_CORE_EXPORT Span2<DataType> view(Int32 rank) const;

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
  ARCANE_CORE_EXPORT Span<DataType> operator()(Int32 rank, Int32 notlocal_id);

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un changement
   * dans le maillage et/ou après un redimensionnement de la variable.
   *
   * Appel collectif.
   */
  ARCANE_CORE_EXPORT void updateVariable();

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

/*!
 * \brief Classe permettant d'accéder aux éléments partagés de la variable
 * en mémoire partagée.
 *
 * Cette classe ne peut pas être utilisée directement. Il est nécessaire
 * d'utiliser une des classes suivante :
 * - \a MachineShMemWinMeshMDVariableT pour les variables au maillage de type
 *   scalaire et de dimension max de 3,
 * - \a MachineShMemWinMeshVectorMDVariableT pour les variables au maillage de
 *   type vecteur et de dimension max de 2,
 * - \a MachineShMemWinMeshMatrixMDVariableT pour les variables au maillage de
 *   type matrice et de dimension max de 1.
 */
template <class ItemType, class DataType, class Extents>
class MachineShMemWinMDVariableT
{

 protected:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  ARCANE_CORE_EXPORT explicit MachineShMemWinMDVariableT(MeshVariableArrayRefT<ItemType, DataType> var);

 public:

  ARCANE_CORE_EXPORT virtual ~MachineShMemWinMDVariableT();

 public:

  /*!
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * Appel non collectif.
   *
   * \return Une vue contenant les ids des rangs.
   */
  ARCANE_CORE_EXPORT ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Méthode permettant d'attendre que tous les processus/threads
   * du noeud appellent cette méthode pour continuer l'exécution.
   */
  ARCANE_CORE_EXPORT void barrier() const;

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

  ARCANE_CORE_EXPORT MDSpan<DataType, typename MDDimType<Extents::rank() + 1>::DimType> view(Int32 rank) const;

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
  ARCANE_CORE_EXPORT MDSpan<DataType, Extents> operator()(Int32 rank, Int32 notlocal_id);

  /*!
   * \brief Méthode permettant de mettre à jour cet objet après un changement
   * dans le maillage et/ou après un redimensionnement de la variable.
   *
   * Appel collectif.
   */
  ARCANE_CORE_EXPORT void updateVariable();

 private:

  Ref<MachineShMemWinVariableMDBase> m_base;
  MeshVariableArrayRefT<ItemType, DataType> m_vart;
  ConstArrayView<Int64> m_nb_elem_dim1;
  Int32 m_nb_elem_dim2{};
  std::array<Int32, Extents::rank()> m_shape_dim2{};
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
 * Cette classe fonctionne pour les variables au maillage de type scalaire et
 * de dimension max de 3.
 *
 * Si le maillage et/ou la taille de la variable change lorsqu'un objet de ce
 * type est utilisé, il est nécessaire d'appeler la méthode \a updateVariable().
 */
template <class ItemType, class DataType, class Extents>
class MachineShMemWinMeshMDVariableT
: public MachineShMemWinMDVariableT<ItemType, DataType, Extents>
{

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  explicit MachineShMemWinMeshMDVariableT(MeshMDVariableRefT<ItemType, DataType, Extents> var)
  : MachineShMemWinMDVariableT<ItemType, DataType, Extents>(var.underlyingVariable())
  {}

  ~MachineShMemWinMeshMDVariableT() override = default;
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
 * Cette classe fonctionne pour les variables au maillage de type vecteur et
 * de dimension max de 2.
 *
 * Si le maillage et/ou la taille de la variable change lorsqu'un objet de ce
 * type est utilisé, il est nécessaire d'appeler la méthode \a updateVariable().
 */
template <class ItemType, class DataType, class Extents>
class MachineShMemWinMeshVectorMDVariableT
: public MachineShMemWinMDVariableT<ItemType, DataType, typename Extents::template AddedFirstExtentsType<DynExtent>>
{
  using AddedFirstExtentsType = Extents::template AddedFirstExtentsType<DynExtent>;

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  template <Int32 Size>
  explicit MachineShMemWinMeshVectorMDVariableT(MeshVectorMDVariableRefT<ItemType, DataType, Size, Extents> var)
  : MachineShMemWinMDVariableT<ItemType, DataType, AddedFirstExtentsType>(var.underlyingVariable())
  {}

  ~MachineShMemWinMeshVectorMDVariableT() override = default;
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
 * Cette classe fonctionne pour les variables au maillage de type matrice et
 * de dimension max de 1.
 *
 * Si le maillage et/ou la taille de la variable change lorsqu'un objet de ce
 * type est utilisé, il est nécessaire d'appeler la méthode \a updateVariable().
 */
template <class ItemType, class DataType, class Extents>
class MachineShMemWinMeshMatrixMDVariableT
: public MachineShMemWinMDVariableT<ItemType, DataType, typename Extents::template AddedFirstLastExtentsType<DynExtent, DynExtent>>
{
  using AddedFirstLastExtentsType = Extents::template AddedFirstLastExtentsType<DynExtent, DynExtent>;

 public:

  /*!
   * \brief Constructeur.
   * \param var Variable ayant la propriété "IVariable::PInShMem".
   */
  template <Int32 Row, Int32 Column>
  explicit MachineShMemWinMeshMatrixMDVariableT(MeshMatrixMDVariableRefT<ItemType, DataType, Row, Column, Extents> var)
  : MachineShMemWinMDVariableT<ItemType, DataType, AddedFirstLastExtentsType>(var.underlyingVariable())
  {}

  ~MachineShMemWinMeshMatrixMDVariableT() override = default;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
