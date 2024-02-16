// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVector.h                                       (C) 2000-2024 */
/*                                                                           */
/* Vecteur sur des entités composants.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_COMPONENTITEMVECTOR_H
#define ARCANE_CORE_MATERIALS_COMPONENTITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/AutoRef.h"

#include "arcane/ItemGroup.h"

#include "arcane/core/materials/IMeshComponent.h"
#include "arcane/core/materials/ComponentItemVectorView.h"
#include "arcane/core/materials/ComponentPartItemVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class ConstituentItemLocalIdList;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur sur les entités d'un composant.
 *
 * \warning Ce vecteur n'est valide que tant que le milieu et le groupe support
 * ne change pas.
 *
 * Cette classe est similaire à la classe ItemVector mais contient une liste
 * d'entités d'un composant (IMeshComponent). Toutes les entités doivent
 * appartenir au même composant.
 *
 * Cette classe utilise une sémantique par référence. Pour effectuer une copie,
 * il faut utiliser la commande clone() ou construire un objet via une vue:
 *
 \code
 * ComponentItemVector v1 = ...;
 * ComponentItemVector v2 = v1; // v2 fait référence à v1
 * ComponentItemVector v3 = v1.clone(); // v3 est une copie de v1
 * ComponentItemVector v4 = v1.view(); // v4 est une copie de v1
 \endcode
 */
class ARCANE_CORE_EXPORT ComponentItemVector
{
  friend class EnvCellVector;
  friend class MatCellVector;

  /*!
   * \brief Implémentation de ComponentItemVector.
   */
  class Impl : public SharedReference
  {
   public:

    Impl(IMeshComponent* component);
    Impl(IMeshComponent* component, ConstArrayView<ComponentItemInternal*> items_internal,
         ConstArrayView<MatVarIndex> matvar_indexes, ConstArrayView<Int32> items_local_id);

   private:

    Impl(const Impl& rhs) = delete;
    Impl(Impl&& rhs) = delete;
    Impl& operator=(const Impl& rhs) = delete;

   public:

    void deleteMe() override;

   public:

    IMeshMaterialMng* m_material_mng = nullptr;
    IMeshComponent* m_component = nullptr;
    UniqueArray<MatVarIndex> m_matvar_indexes;
    UniqueArray<Int32> m_items_local_id;
    std::unique_ptr<MeshComponentPartData> m_part_data;
    std::unique_ptr<ConstituentItemLocalIdList> m_constituent_list;
  };

 public:

  //! Constructeur de recopie. Cette instance fait ensuite référence à \a rhs
  ComponentItemVector(const ComponentItemVector& rhs) = default;

  //! Opérateur de recopie
  ComponentItemVector& operator=(const ComponentItemVector&) = default;

 protected:

  //! Construit un vecteur pour le composant \a component
  explicit ComponentItemVector(IMeshComponent* component);
  //! Constructeur de recopie. Cette instance est une copie de \a rhs.
  explicit ComponentItemVector(ComponentItemVectorView rhs);

 public:

  //! Conversion vers une vue sur ce vecteur
  operator ComponentItemVectorView() const
  {
    return view();
  }

  //! Vue sur ce vecteur
  ComponentItemVectorView view() const;

  //! Composant associé
  IMeshComponent* component() const { return m_p->m_component; }

  //! Clone ce vecteur
  ComponentItemVector clone() const { return ComponentItemVector(view()); }

 public:

  //! Liste des entités pures (associées à la maille globale) du composant
  ComponentPurePartItemVectorView pureItems() const;
  //! Liste des entités impures (partielles) du composant
  ComponentImpurePartItemVectorView impureItems() const;

 private:

  ConstArrayView<MatVarIndex> _matvarIndexes() const { return m_p->m_matvar_indexes; }
  ConstituentItemLocalIdListView _constituentItemListView() const;

  ARCANE_DEPRECATED_REASON("Use overload with ComponentItemInternalLocalId instead")
  ConstArrayView<ComponentItemInternal*> _itemsInternalView() const;

 protected:

  ARCANE_DEPRECATED_REASON("Use overload with ComponentItemInternalLocalId instead")
  void _setItemsInternal(ConstArrayView<ComponentItemInternal*> globals,
                         ConstArrayView<ComponentItemInternal*> multiples);

  void _setMatVarIndexes(ConstArrayView<MatVarIndex> globals,
                         ConstArrayView<MatVarIndex> multiples);
  void _setLocalIds(ConstArrayView<Int32> globals, ConstArrayView<Int32> multiples);
  ConstArrayView<Int32> _localIds() const
  {
    return m_p->m_items_local_id.constView();
  }

  IMeshMaterialMng* _materialMng() const { return m_p->m_material_mng; }
  IMeshComponent* _component() const { return m_p->m_component; }

 private:

  AutoRefT<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

