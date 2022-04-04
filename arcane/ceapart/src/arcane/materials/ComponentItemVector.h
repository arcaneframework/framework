﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVector.h                                       (C) 2000-2017 */
/*                                                                           */
/* Vecteur sur des entités composants.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_COMPONENTITEMVECTOR_H
#define ARCANE_MATERIALS_COMPONENTITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/AutoRef.h"

#include "arcane/ItemGroup.h"

#include "arcane/materials/IMeshComponent.h"
#include "arcane/materials/MeshMaterialVariableIndexer.h"
#include "arcane/materials/ComponentItemVectorView.h"
#include "arcane/materials/ComponentPartItemVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;
class IMeshComponent;
class ComponentItemVectorView;
class MeshComponentPartData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
class ARCANE_MATERIALS_EXPORT ComponentItemVector
{
  /*!
   * \brief Implémentation de ComponentItemVector.
   */
  class Impl: public SharedReference
  {
   public:
    Impl(IMeshComponent* component);
    Impl(Impl&& rhs);
    Impl(IMeshComponent* component,ConstArrayView<ComponentItemInternal*> items_internal,
         ConstArrayView<MatVarIndex> matvar_indexes);
   protected:
    ~Impl() override;
   public:
    void deleteMe() override;
   public:
    IMeshMaterialMng* m_material_mng;
    IMeshComponent* m_component;
    UniqueArray<ComponentItemInternal*> m_items_internal;
    UniqueArray<MatVarIndex> m_matvar_indexes;
    MeshComponentPartData* m_part_data;
  };

 public:

  //! Constructeur de recopie. Cette instance fait ensuite référence à \a rhs
  ComponentItemVector(const ComponentItemVector& rhs) = default;

  //! Opérateur de recopie
  ComponentItemVector& operator=(const ComponentItemVector&) = default;

 protected:

  //! Construit un vecteur pour le composant \a component
  ComponentItemVector(IMeshComponent* component);
  //! Constructeur de recopie. Cette instance est une copie de \a rhs.
  ComponentItemVector(ComponentItemVectorView rhs);

 public:

  //! Conversion vers une vue sur ce vecteur
  operator ComponentItemVectorView() const
  { return view(); }

  //! Vue sur ce vecteur
  ComponentItemVectorView view() const
  {
    return ComponentItemVectorView(m_p->m_component,m_p->m_matvar_indexes,m_p->m_items_internal);
  }

  //! Composant associé
  IMeshComponent* component() const { return m_p->m_component; }

  //! Clone ce vecteur
  ComponentItemVector clone() const { return ComponentItemVector(view()); }

 public:

  //! Liste des entités pures (associées à la maille globale) du composant
  ComponentPurePartItemVectorView pureItems() const;
  //! Liste des entités impures (partielles) du composant
  ComponentImpurePartItemVectorView impureItems() const;

 public:

  //! Interne à Arcane
  //@{
  ConstArrayView<MatVarIndex> matvarIndexes() const { return m_p->m_matvar_indexes; }
  ConstArrayView<ComponentItemInternal*> itemsInternalView() const
  {
    return m_p->m_items_internal.constView();
  }
  //@}

 protected:

  void _setItemsInternal(ConstArrayView<ComponentItemInternal*> globals,
                         ConstArrayView<ComponentItemInternal*> multiples);

  void _setMatVarIndexes(ConstArrayView<MatVarIndex> globals,
                         ConstArrayView<MatVarIndex> multiples);

  IMeshMaterialMng* _materialMng() const { return m_p->m_material_mng; }
  IMeshComponent* _component() const { return m_p->m_component; }

 private:

  AutoRefT<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

