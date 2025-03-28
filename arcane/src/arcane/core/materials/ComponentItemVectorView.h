// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVectorView.h                                   (C) 2000-2024 */
/*                                                                           */
/* Vue sur un vecteur sur des entités de constituants.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_COMPONENTITEMVECTORVIEW_H
#define ARCANE_CORE_MATERIALS_COMPONENTITEMVECTORVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemGroup.h"

#include "arcane/core/materials/MatVarIndex.h"
#include "arcane/core/materials/IMeshComponent.h"
#include "arcane/core/materials/ComponentItemInternal.h"
#include "arcane/core/materials/MatItem.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
class MeshMaterialTesterModule;
class MaterialHeatTestModule;
}
namespace Arcane::Accelerator::impl
{
class ConstituentCommandContainerBase;
}

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un vecteur sur les entités d'un composant.
 *
 * Les constructeurs de cette classe sont internes à %Arcane.
 */
class ARCANE_CORE_EXPORT ComponentItemVectorView
{
  friend class ComponentItemVector;
  friend class ConstituentItemVectorImpl;
  friend class MatItemVectorView;
  friend class EnvItemVectorView;
  friend class MatCellEnumerator;
  friend class EnvCellEnumerator;
  friend class ComponentCellEnumerator;
  friend Arcane::Accelerator::impl::ConstituentCommandContainerBase;
  friend ArcaneTest::MeshMaterialTesterModule;
  friend ArcaneTest::MaterialHeatTestModule;
  template <typename ViewType, typename LambdaType>
  friend class LambdaMatItemRangeFunctorT;
  template <typename DataType> friend class
  MaterialVariableArrayTraits;

 public:

  ComponentItemVectorView() = default;

 protected:

  //! Construit un vecteur contenant les entités de \a group pour le composant \a component
  ComponentItemVectorView(IMeshComponent* component,
                          ConstArrayView<MatVarIndex> mvi,
                          ConstituentItemLocalIdListView constituent_local_ids,
                          ConstArrayView<Int32> local_ids)
  : m_matvar_indexes_view(mvi)
  , m_constituent_list_view(constituent_local_ids)
  , m_items_local_id_view(local_ids)
  , m_component(component)
  {
  }

  //! Construit une vue vide pour le composant \a component
  explicit ComponentItemVectorView(IMeshComponent* component)
  : m_component(component)
  {
  }

  //! Construit une vue à partir d'une autre vue.
  ComponentItemVectorView(IMeshComponent* component, ComponentItemVectorView rhs_view)
  : m_matvar_indexes_view(rhs_view.m_matvar_indexes_view)
  , m_constituent_list_view(rhs_view.m_constituent_list_view)
  , m_items_local_id_view(rhs_view.m_items_local_id_view)
  , m_component(component)
  {
  }

 public:

  //! Nombre d'entités dans la vue
  Integer nbItem() const { return m_matvar_indexes_view.size(); }

  //! Composant associé
  IMeshComponent* component() const { return m_component; }

  //! Retourne la \a index-ème ComponentCell de la vue
  ARCCORE_HOST_DEVICE ComponentCell componentCell(Int32 index) const
  {
    return m_constituent_list_view._constituenItemBase(index);
  }

 private:

  // Tableau des MatVarIndex de cette vue.
  ConstArrayView<MatVarIndex> _matvarIndexes() const { return m_matvar_indexes_view; }

  //! Tableau des localId() des entités associées
  ConstArrayView<Int32> _internalLocalIds() const { return m_items_local_id_view; }

  ConstituentItemLocalIdListView _constituentItemListView() const { return m_constituent_list_view; }
  /*!
   * \internal
   * \brief Créé une sous-vue de cette vue.
   * 
   * Cette méthode est interne à Arcane et ne doit pas être utilisée.
   */
  ComponentItemVectorView _subView(Integer begin, Integer size);

  //! Pour les tests vérifie que \a rhs et l'instance pointent sur les même données
  bool _isSamePointerData(const ComponentItemVectorView& rhs) const
  {
    bool test1 = m_constituent_list_view._isSamePointerData(rhs.m_constituent_list_view);
    return test1 && (m_matvar_indexes_view.data() == rhs.m_matvar_indexes_view.data());
  }

 private:

  // NOTE: Cette classe est wrappée directement en C#.
  // Si on modifie les champs de cette classe, il faut modifier le type correspondant
  // dans le wrappeur.
  ConstArrayView<MatVarIndex> m_matvar_indexes_view;
  ConstituentItemLocalIdListView m_constituent_list_view;
  ConstArrayView<Int32> m_items_local_id_view;
  IMeshComponent* m_component = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un vecteur sur les entités d'un matériau.
 *
 * Les constructeurs de cette classe sont internes à %Arcane.
 */
class ARCANE_CORE_EXPORT MatItemVectorView
: public ComponentItemVectorView
{
  friend class MatCellVector;
  friend class MeshMaterial;
  template <typename ViewType, typename LambdaType>
  friend class LambdaMatItemRangeFunctorT;

 public:

  MatItemVectorView() = default;

 private:

  MatItemVectorView(IMeshComponent* component,
                    ConstArrayView<MatVarIndex> mv_indexes,
                    ConstituentItemLocalIdListView constituent_local_ids,
                    ConstArrayView<Int32> local_ids)
  : ComponentItemVectorView(component, mv_indexes, constituent_local_ids, local_ids)
  {}

  MatItemVectorView(IMeshComponent* component, ComponentItemVectorView v)
  : ComponentItemVectorView(component, v)
  {}

 private:

  /*!
   * \internal
   * \brief Créé une sous-vue de cette vue.
   * 
   * Cette méthode est interne à Arcane et ne doit pas être utilisée.
   */
  MatItemVectorView _subView(Integer begin, Integer size);

 public:

  //! Matériau associé
  IMeshMaterial* material() const;

  //! Récupère la index-ème MatCell de la vue
  ARCCORE_HOST_DEVICE MatCell matCell(Int32 index) const { return MatCell(componentCell(index)); }

  // Temporaire: à conserver pour compatibilité
  ARCANE_DEPRECATED_240 MatItemVectorView subView(Integer begin, Integer size)
  {
    return _subView(begin, size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un vecteur sur les entités d'un milieu.
 *
 * Les constructeurs de cette classe sont internes à %Arcane.
 */
class ARCANE_CORE_EXPORT EnvItemVectorView
: public ComponentItemVectorView
{
  friend class EnvCellVector;
  friend class MeshEnvironment;
  template <typename ViewType, typename LambdaType>
  friend class LambdaMatItemRangeFunctorT;

 public:

  EnvItemVectorView() = default;

 private:

  EnvItemVectorView(IMeshComponent* component,
                    ConstArrayView<MatVarIndex> mv_indexes,
                    ConstituentItemLocalIdListView constituent_local_ids,
                    ConstArrayView<Int32> local_ids)
  : ComponentItemVectorView(component, mv_indexes, constituent_local_ids, local_ids)
  {}

  EnvItemVectorView(IMeshComponent* component, ComponentItemVectorView v)
  : ComponentItemVectorView(component, v)
  {}

 private:

  /*!
   * \internal
   * \brief Créé une sous-vue de cette vue.
   * 
   * Cette méthode est interne à Arcane et ne doit pas être utilisée.
   */
  EnvItemVectorView _subView(Integer begin, Integer size);

 public:

  //! Milieu associé
  IMeshEnvironment* environment() const;

  //! Récupère la index-ème EnvCell de la vue
  ARCCORE_HOST_DEVICE EnvCell envCell(Int32 index) const { return EnvCell(componentCell(index)); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

