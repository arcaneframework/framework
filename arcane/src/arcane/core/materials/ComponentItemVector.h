// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVector.h                                       (C) 2000-2025 */
/*                                                                           */
/* Vector over constituent entities.                                         */
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
 * \brief Interface for the implementation of ComponentItemVector.
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
 * \brief Vector over the entities of a constituent.
 *
 * \warning This vector is only valid as long as the medium and the supporting group
 * do not change.
 *
 * This class is similar to the ItemVector class but contains a list
 * of constituent entities (IMeshComponent). All entities must
 * belong to the same constituent.
 *
 * Instances of this class are generally not constructed
 * directly unless they are copies. For original creation, MatCellVector or EnvCellVector must be used.
 *
 * This class uses a reference semantics. To perform a copy,
 * you must use the clone() command or construct an object via a view:
 *
 \code
 * ComponentItemVector v1 = ...;
 * ComponentItemVector v2 = v1; // v2 references v1
 * ComponentItemVector v3 = v1.clone(); // v3 is a copy of v1
 * ComponentItemVector v4 = v1.view(); // v4 is a copy of v1
 \endcode
 */
class ARCANE_CORE_EXPORT ComponentItemVector
{
 public:

  //! Copy constructor. This instance then references \a rhs
  ComponentItemVector(const ComponentItemVector& rhs) = default;

  //! Copy assignment operator
  ComponentItemVector& operator=(const ComponentItemVector&) = default;

 protected:

  //! Constructs a vector for the constituent \a component
  explicit ComponentItemVector(IMeshComponent* component);

  //! Copy constructor. This instance is a copy of \a rhs.
  explicit ComponentItemVector(ComponentItemVectorView rhs);

 public:

  //! Conversion to a view of this vector
  operator ComponentItemVectorView() const
  {
    return view();
  }

  //! View of this vector
  ComponentItemVectorView view() const;

  //! Associated constituent
  IMeshComponent* component() const { return _component(); }

  //! Clone this vector
  ComponentItemVector clone() const { return ComponentItemVector(view()); }

 public:

  //! List of pure entities (associated with the global mesh) of the constituent
  ComponentPurePartItemVectorView pureItems() const;

  //! List of impure (partial) entities of the constituent
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
