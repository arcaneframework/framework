// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMDVariableRef.h                                         (C) 2000-2022 */
/*                                                                           */
/* Classe gérant une variable multi-dimension sur une entité du maillage.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHMDVARIABLEREF_H
#define ARCANE_MESHMDVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayLayout.h"
#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/MDSpan.h"

#include "arcane/MeshVariableArrayRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * ATTENTION:
 *
 * Toutes les classes de ce fichier sont expérimentales et l'API n'est pas
 * figée. A NE PAS UTILISER EN DEHORS DE ARCANE.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType>
class MeshMDVariableRefWrapperT
: public MeshVariableArrayRefT<ItemType, DataType>
{
  template <typename _ItemType, typename _DataType, typename _Extents>
  friend class Arcane::MeshMDVariableRefBaseT;

 public:

  using BaseClass = MeshVariableArrayRefT<ItemType, DataType>;
  using VariableType = typename BaseClass::PrivatePartType;
  using ValueDataType = typename VariableType::ValueDataType;

 private:

  explicit MeshMDVariableRefWrapperT(const VariableBuildInfo& vbi)
  : BaseClass(vbi)
  {
  }

 private:

  ValueDataType* trueData() { return this->m_private_part->trueData(); }

  void fillShape(ArrayShape& shape_with_item)
  {
    this->m_private_part->fillShape(shape_with_item);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base gérant une variable multi-dimension sur une entité du maillage.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane.
 */
template <typename ItemType, typename DataType, typename Extents>
class MeshMDVariableRefBaseT
: public MeshVariableRef
{
 public:

  using UnderlyingVariableType = MeshVariableArrayRefT<ItemType, DataType>;
  using MDSpanType = MDSpan<DataType, Extents, RightLayout>;
  using ItemLocalIdType = typename ItemType::LocalIdType;

 public:

  explicit MeshMDVariableRefBaseT(const VariableBuildInfo& b)
  : MeshVariableRef()
  , m_underlying_var(b)
  {
    _internalInit(m_underlying_var.variable());
  }

  UnderlyingVariableType& underlyingVariable() { return m_underlying_var; }

 protected:

  void updateFromInternal() override
  {
    const Int32 nb_rank = Extents::rank();
    ArrayShape shape_with_item;
    shape_with_item.setNbDimension(nb_rank);
    m_underlying_var.fillShape(shape_with_item);

    ArrayExtents<Extents> new_extents = ArrayExtentsBase<Extents>::fromSpan(shape_with_item.dimensions());
    m_mdspan = MDSpanType(m_underlying_var.trueData()->view().data(), new_extents);
  }

 protected:

  impl::MeshMDVariableRefWrapperT<ItemType, DataType> m_underlying_var;
  MDSpanType m_mdspan;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant une variable multi-dimension sur une entité du maillage.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane.
 */
template <typename ItemType, typename DataType, typename Extents>
class MeshMDVariableRefT
: public MeshMDVariableRefBaseT<ItemType, DataType, typename Extents::template AddedFirstExtentsType<DynExtent>>
{
  using AddedFirstExtentsType = typename Extents::template AddedFirstExtentsType<DynExtent>;
  static_assert(Extents::rank() >= 1 && Extents::rank() <= 3, "Only Extents of rank 1, 2 or 3 are implemented");

 public:

  using BaseClass = MeshMDVariableRefBaseT<ItemType, DataType, AddedFirstExtentsType>;
  using ItemLocalIdType = typename ItemType::LocalIdType;

 public:

  explicit MeshMDVariableRefT(const VariableBuildInfo& b)
  : BaseClass(b)
  {}

 public:

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  DataType& operator()(ItemLocalIdType id, Int32 i1)
  {
    return this->m_mdspan(id.localId(), i1);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  const DataType& operator()(ItemLocalIdType id, Int32 i1) const
  {
    return this->m_mdspan(id.localId(), i1);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  DataType& operator()(ItemLocalIdType id, Int32 i1, Int32 i2)
  {
    return this->m_mdspan(id.localId(), i1, i2);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  const DataType& operator()(ItemLocalIdType id, Int32 i1, Int32 i2) const
  {
    return this->m_mdspan(id.localId(), i1, i2);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 3, void>>
  DataType& operator()(ItemLocalIdType id, Int32 i, Int32 j, Int32 k)
  {
    return this->m_mdspan(id.localId(), i, j, k);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 3, void>>
  const DataType& operator()(ItemLocalIdType id, Int32 i, Int32 j, Int32 k) const
  {
    return this->m_mdspan(id.localId(), i, j, k);
  }

  /*!
   * \brief Change la forme de la donnée.
   *
   * Le nombre d'éléments de \a dims doit correspondre aux nombre de valeurs
   * dynamiques de \a Extents.
   */
  void reshape(std::array<Int32, Extents::nb_dynamic> dims)
  {
    ArrayShape shape(dims);
    this->m_underlying_var.resizeAndReshape(shape);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
