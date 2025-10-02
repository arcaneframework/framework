// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVector.h                                       (C) 2000-2025 */
/*                                                                           */
/* Vecteur sur des entités constituants.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_COMPONENTITEMVECTOR_H
#define ARCANE_CORE_MATERIALS_COMPONENTITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Ref.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/core/ItemGroup.h"

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
 * \brief Interface pour l'implémentation de ComponentItemVector.
 */
class IConstituentItemVectorImpl
{
  friend class ComponentItemVector;
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  virtual ~IConstituentItemVectorImpl() = default;

 protected:

  virtual ComponentItemVectorView _view() const = 0;
  virtual ComponentPurePartItemVectorView _pureItems() const = 0;
  virtual ComponentImpurePartItemVectorView _impureItems() const = 0;
  virtual ConstArrayView<Int32> _localIds() const = 0;
  virtual IMeshMaterialMng* _materialMng() const = 0;
  virtual IMeshComponent* _component() const = 0;
  virtual ConstArrayView<MatVarIndex> _matvarIndexes() const = 0;
  virtual ConstituentItemLocalIdListView _constituentItemListView() const = 0;
  virtual void _setItems(SmallSpan<const Int32> local_ids) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur sur les entités d'un constituant.
 *
 * \warning Ce vecteur n'est valide que tant que le milieu et le groupe support
 * ne change pas.
 *
 * Cette classe est similaire à la classe ItemVector mais contient une liste
 * d'entités d'un constituant (IMeshComponent). Toutes les entités doivent
 * appartenir au même constituant.
 *
 * Les instances de cette classe ne sont en général pas construit
 * directement sauf s'il s'agit de copies. Pour la création d'origine, il
 * faut utilise MatCellVector ou EnvCellVector.
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
 public:

  //! Constructeur de recopie. Cette instance fait ensuite référence à \a rhs
  ComponentItemVector(const ComponentItemVector& rhs) = default;

  //! Opérateur de recopie
  ComponentItemVector& operator=(const ComponentItemVector&) = default;

 protected:

  //! Construit un vecteur pour le constituant \a component
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

  //! Constituant associé
  IMeshComponent* component() const { return _component(); }

  //! Clone ce vecteur
  ComponentItemVector clone() const { return ComponentItemVector(view()); }

 public:

  //! Liste des entités pures (associées à la maille globale) du constituant
  ComponentPurePartItemVectorView pureItems() const;
  //! Liste des entités impures (partielles) du constituant
  ComponentImpurePartItemVectorView impureItems() const;

 protected:

  ConstArrayView<MatVarIndex> _matvarIndexes() const;
  ConstituentItemLocalIdListView _constituentItemListView() const;

 protected:

  void _setItems(SmallSpan<const Int32> local_ids);
  ConstArrayView<Int32> _localIds() const;
  IMeshMaterialMng* _materialMng() const;
  IMeshComponent* _component() const;

 private:

  Ref<IConstituentItemVectorImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

