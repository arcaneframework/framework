// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdItem.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Types of entities and entity enumerators for vectorization.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SIMDITEM_H
#define ARCANE_CORE_SIMDITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Simd.h"

// The ARCANE_SIMD_BENCH macro is only defined for the Simd
// Simd (in contribs/Simd) bench and prevents including entity management.

#ifndef ARCANE_SIMD_BENCH
#include "arcane/core/ItemEnumerator.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file SimdItem.h
 *
 * This file contains the type declarations for managing
 * vectorization with mesh entities (Item).
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

template<typename ItemType>
class SimdItemEnumeratorT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * TODO:
 * - Create a version of SimdItem by vector size (2, 4, 8).
 * - Use a mask if possible.
 * - aligned SimdItemBase
 * - create a version of the SimdItemBase constructor without (nb_valid)
 * for the case where the vector is full.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Class managing a SIMD vector of entities.
 *
 * This class stores \a N mesh entities, where \a N depends
 * on the size of the SIMD registers and equals SimdInfo::Int32IndexSize.
 *
 * This class is not used directly. SimdItem or
 * SimdItemT must be used.
 */
class ARCANE_CORE_EXPORT SimdItemBase
{
 protected:
  
  typedef ItemInternal* ItemInternalPtr;

 public:

 typedef SimdInfo::SimdInt32IndexType SimdIndexType;

 public:

 /*!
   * \brief Constructs an instance.
   * \warning \a ids must have the required alignment for a SimdIndexType.
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use another constructor")
  SimdItemBase(const ItemInternalPtr* items, const SimdIndexType* ids)
  : m_simd_local_ids(*ids), m_shared_info(ItemInternalCompatibility::_getSharedInfo(items,1)) { }

 protected:

  SimdItemBase(ItemSharedInfo* shared_info,const SimdIndexType* ids)
  : m_simd_local_ids(*ids), m_shared_info(shared_info) { }

 public:

  //! Internal part (for internal use only)
  ARCANE_DEPRECATED_REASON("Y2022: Use method SimdItem::item() instead")
  ItemInternal* item(Integer si) const { return m_shared_info->m_items_internal[localId(si)]; }

  ARCANE_DEPRECATED_REASON("Y2022: Use method SimdItem::operator[]() instead")
  ItemInternal* operator[](Integer si) const { return m_shared_info->m_items_internal[localId(si)]; }

  //! List of local IDs of the instance entities
  const SimdIndexType& ARCANE_RESTRICT simdLocalIds() const { return m_simd_local_ids; }

  //! List of local IDs of the instance entities
  const Int32* ARCANE_RESTRICT localIds() const { return (const Int32*)&m_simd_local_ids; }

  //! Local ID of the entity at index \a index.
  Int32 localId(Int32 index) const { return m_simd_local_ids[index]; }

 protected:

  SimdIndexType m_simd_local_ids;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimdItemDirectBase
{
 protected:

  typedef ItemInternal* ItemInternalPtr;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use another constructor")
  SimdItemDirectBase(const ItemInternalPtr* items,Int32 base_local_id,Integer nb_valid)
  : m_base_local_id(base_local_id), m_nb_valid(nb_valid),
    m_shared_info(ItemInternalCompatibility::_getSharedInfo(items,nb_valid)) { }

 protected:

  SimdItemDirectBase(ItemSharedInfo* shared_info,Int32 base_local_id,Integer nb_valid)
  : m_base_local_id(base_local_id), m_nb_valid(nb_valid), m_shared_info(shared_info) {}

  // TEMPORARY to avoid deprecated
  SimdItemDirectBase(Int32 base_local_id,Integer nb_valid,const ItemInternalPtr* items)
  : m_base_local_id(base_local_id), m_nb_valid(nb_valid),
    m_shared_info(ItemInternalCompatibility::_getSharedInfo(items,nb_valid)) { }

 public:

  //! Number of valid entities in the instance.
  inline Integer nbValid() const { return m_nb_valid; }

  //! List of local IDs of the instance entities
  inline Int32 baseLocalId() const { return m_base_local_id; }

 protected:

  Int32 m_base_local_id;
  Integer m_nb_valid;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vector index with indirection for an entity type.
 * TODO: store the indices in a vector register to be able
 * to perform the gather quickly. For this, create the equivalent of AVXSimdReal
 * for Int32.
 */
template<typename ItemType>
class SimdItemIndexT
{
 public:
  typedef SimdInfo::SimdInt32IndexType SimdIndexType;
 public:
  SimdItemIndexT(const SimdIndexType& ARCANE_RESTRICT local_ids)
  : m_local_ids(local_ids){}
  SimdItemIndexT(const SimdIndexType* ARCANE_RESTRICT local_ids)
  : m_local_ids(*local_ids){}
 public:
  //! List of local IDs of the instance entities
  const SimdIndexType& ARCANE_RESTRICT simdLocalIds() const { return m_local_ids; }
 private:
  const SimdIndexType& ARCANE_RESTRICT m_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vector index without indirection for an entity type
 */
template<typename ItemType>
class SimdItemDirectIndexT
{
 public:
  SimdItemDirectIndexT(Int32 base_local_id)
  : m_base_local_id(base_local_id){}
 public:
  inline Int32 baseLocalId() const { return m_base_local_id; }
 private:
  Int32 m_base_local_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Manages a vector of \a Item entities.
 */
class SimdItem
: public SimdItemBase
{
 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use another constructor")
  SimdItem(const ItemInternalPtr* items,const SimdInfo::SimdInt32IndexType* ids)
  : SimdItemBase(ItemInternalCompatibility::_getSharedInfo(items,1),ids) { }

 protected:

  SimdItem(ItemSharedInfo* shared_info,const SimdInfo::SimdInt32IndexType* ids)
  : SimdItemBase(shared_info,ids) { }

 public:

  //! inline \a si-th entity of the instance
  inline Item item(Int32 si) const { return Item(localId(si),m_shared_info); }

  //! inline \a si-th entity of the instance
  inline Item operator[](Int32 si) const { return Item(localId(si),m_shared_info); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Manages a vector of \a ItemType entities.
 */
template<typename ItemType>
class SimdItemT
: public SimdItem
{
  friend class SimdItemEnumeratorT<ItemType>;

 protected:
  
  typedef ItemInternal* ItemInternalPtr;

 public:

#if 0
  ARCANE_DEPRECATED_REASON("Y2022: Use another constructor")
  SimdItemT(const ItemInternalPtr* items,const SimdInfo::SimdInt32IndexType* ids)
  : SimdItem(items,ids) { }
#endif

 private:

  SimdItemT(ItemSharedInfo* shared_info,const SimdInfo::SimdInt32IndexType* ids)
  : SimdItem(shared_info,ids) { }

 public:

  //! Returns the \a si-th entity of the instance
  ItemType item(Integer si) const
  {
    return ItemType(localId(si),m_shared_info);
  }

  //! Returns the \a si-th entity of the instance
  ItemType operator[](Integer si) const
  {
    return ItemType(localId(si),m_shared_info);
  }

  operator SimdItemIndexT<ItemType>()
  {
    return SimdItemIndexT<ItemType>(this->simdLocalIds());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Manages a vector of \a ItemType entities.
 */
template<typename ItemType>
class SimdItemDirectT
: public SimdItemDirectBase
{
  friend class SimdItemEnumeratorT<ItemType>;

 protected:

  typedef ItemInternal* ItemInternalPtr;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use another constructor")
  SimdItemDirectT(const ItemInternalPtr* items,Int32 base_local_id,Integer nb_valid)
  : SimdItemDirectBase(base_local_id,nb_valid,items) {}

 private:

  SimdItemDirectT(ItemSharedInfo* shared_info,Int32 base_local_id,Integer nb_valid)
  : SimdItemDirectBase(shared_info,base_local_id,nb_valid) {}

 public:

  operator SimdItemDirectIndexT<ItemType>()
  {
    return SimdItemDirectIndexT<ItemType>(this->m_base_local_id);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \ingroup ArcaneSimd
 * \brief Object allowing positioning of values in a SIMD vector.
 */
template<typename DataType>
class SimdSetter
{
  typedef typename SimdTypeTraits<DataType>::SimdType SimdType;
 public:
  SimdSetter(DataType* ARCANE_RESTRICT _data,
             const SimdInfo::SimdInt32IndexType& ARCANE_RESTRICT _indexes)
  : idx(_indexes), m_data(_data)
  {
  }
 public:
  void operator=(const SimdType& vr)
  {
    vr.set(m_data,idx);
  }
  void operator=(const DataType& v)
  {
    SimdType vr(v);
    vr.set(m_data,idx);
  }
 private:
  const SimdInfo::SimdInt32IndexType& ARCANE_RESTRICT idx;
  DataType* ARCANE_RESTRICT m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Object allowing positioning of values in a SIMD vector.
 */
template<typename DataType>
class SimdDirectSetter
{
  typedef typename SimdTypeTraits<DataType>::SimdType SimdType;
 public:
  SimdDirectSetter(DataType* ARCANE_RESTRICT _data)
  : m_data(_data) { }
 public:
  void operator=(const SimdType& vr)
  {
    vr.set(m_data);
  }
 private:
  DataType* ARCANE_RESTRICT m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Base class for enumerators over vectorial entities (SimdItem).
 */
class ARCANE_CORE_EXPORT SimdItemEnumeratorBase
: public SimdEnumeratorBase
{
 protected:
  
  typedef ItemInternal* ItemInternalPtr;
  
 public:

  typedef SimdInfo::SimdInt32IndexType SimdIndexType;

 public:

  // TODO: Handle m_local_id_offset for this class

  // TODO: By end of 2024, make certain constructors internal to Arcane and deprecate others.
  // Do the same for derived classes

  SimdItemEnumeratorBase() = default;

  // TODO: Make internal to Arcane
  SimdItemEnumeratorBase(const ItemInternalVectorView& view)
  : SimdEnumeratorBase(view.localIds()), m_shared_info(view.m_shared_info) {}
  // TODO: Make internal to Arcane
  SimdItemEnumeratorBase(const ItemEnumerator& rhs)
  : SimdEnumeratorBase(rhs.m_view.m_local_ids,rhs.count()), m_shared_info(rhs.m_item.m_shared_info) {}

  // TODO: deprecate
  SimdItemEnumeratorBase(const ItemInternalPtr* items,const Int32* local_ids,Integer n)
  : SimdEnumeratorBase(local_ids,n), m_shared_info(ItemInternalCompatibility::_getSharedInfo(items,n)) { }
  // TODO: deprecate
  SimdItemEnumeratorBase(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids)
  : SimdEnumeratorBase(local_ids), m_shared_info(ItemInternalCompatibility::_getSharedInfo(items.data(),local_ids.size())) { }

 public:

  // TODO: deprecate
  //! List of entities
  const ItemInternalPtr* unguardedItems() const { return m_shared_info->m_items_internal.data(); }

 protected:

  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneSimd
 * \brief Enumerator over a list of entities.
 */
template<typename ItemType>
class SimdItemEnumeratorT
: public SimdItemEnumeratorBase
{
 protected:
  
  typedef ItemInternal* ItemInternalPtr;
  
 public:

  typedef SimdItemT<ItemType> SimdItemType;

  SimdItemEnumeratorT()
  : SimdItemEnumeratorBase(){}
  SimdItemEnumeratorT(const ItemEnumerator& rhs)
  : SimdItemEnumeratorBase(rhs){}
  SimdItemEnumeratorT(const ItemEnumeratorT<ItemType>& rhs)
  : SimdItemEnumeratorBase(rhs){}
  SimdItemEnumeratorT(const ItemVectorViewT<ItemType>& rhs)
  : SimdItemEnumeratorBase(rhs) {}

  // TODO: deprecate
  SimdItemEnumeratorT(const ItemInternalPtr* items,const Int32* local_ids,Integer n)
  : SimdItemEnumeratorBase(items,local_ids,n){}
  // TODO: deprecate
  SimdItemEnumeratorT(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids)
  : SimdItemEnumeratorBase(items,local_ids) {}

 public:

  SimdItemType operator*() const
  {
    return SimdItemType(m_shared_info,_currentSimdIndex());
  }

  SimdItemDirectT<ItemType> direct() const
  {
    return SimdItemDirectT<ItemType>(m_shared_info,m_index,nbValid());
  }

  operator SimdItemIndexT<ItemType>()
  {
    return SimdItemIndexT<ItemType>(_currentSimdIndex());
  }

#ifndef ARCANE_SIMD_BENCH
  inline ItemEnumeratorT<ItemType> enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_shared_info,Int32ConstArrayView(nbValid(),m_local_ids+m_index));
  }
#endif

 protected:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_SIMD_BENCH
/*!
 * \ingroup ArcaneSimd
 * \brief SIMD vector of \a Node.
 */
typedef SimdItemT<Node> SimdNode;
/*!
 * \ingroup ArcaneSimd
 * \brief SIMD vector of \a Edge.
 */
typedef SimdItemT<Edge> SimdEdge;
/*!
 * \ingroup ArcaneSimd
 * \brief SIMD vector of \a Face.
 */
typedef SimdItemT<Face> SimdFace;
/*!
 * \ingroup ArcaneSimd
 * \brief SIMD vector of \a Cell.
 */
typedef SimdItemT<Cell> SimdCell;
/*!
 * \ingroup ArcaneSimd
 * \brief SIMD vector of \a Particle.
 */
typedef SimdItemT<Particle> SimdParticle;
#else
/*!
 * \ingroup ArcaneSimd
 * \brief SIMD vector of \a Cell.
 */
typedef SimdItemT<Cell> SimdCell;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType>
class SimdItemEnumeratorContainerTraits
{
 public:

  static SimdItemEnumeratorT<ItemType> getSimdEnumerator(const ItemGroupT<ItemType>& g)
  {
    return g._simdEnumerator();
  }
  // Create an iterator from an ItemVectorView. This view must have padding
  // equal to the vector size.
  static SimdItemEnumeratorT<ItemType> getSimdEnumerator(const ItemVectorViewT<ItemType>& g)
  {
    return g.enumerator();
  }

  // For compatibility with existing code
  // If we are here, it means that the type 'T' is not an Arcane type.
  // This call should eventually be forbidden (e.g., by end of 2025)
  template <typename T>
  static SimdItemEnumeratorT<ItemType> getSimdEnumerator(const T& g)
  {
    return g.enumerator();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ENUMERATE_SIMD_(type, iname, view) \
  for (A_TRACE_ITEM_ENUMERATOR(SimdItemEnumeratorT<type>) iname(::Arcane::SimdItemEnumeratorContainerTraits<type>::getSimdEnumerator(view) A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname)

// TODO: To be removed. Use ENUMERATE_SIMD_ instead
#define ENUMERATE_SIMD_GENERIC(type, iname, view) \
  ENUMERATE_SIMD_(type,iname,view)

/*!
 * \ingroup ArcaneSimd
 * \brief SIMD enumerator over a group or list of nodes.
 */
#define ENUMERATE_SIMD_NODE(name, group) ENUMERATE_SIMD_(::Arcane::Node, name, group)

/*!
 * \ingroup ArcaneSimd
 * \brief SIMD enumerator over a group or list of edges.
 */
#define ENUMERATE_SIMD_EDGE(name, group) ENUMERATE_SIMD_(::Arcane::Edge, name, group)

/*!
 * \ingroup ArcaneSimd
 * \brief SIMD enumerator over a group or list of faces.
 */
#define ENUMERATE_SIMD_FACE(name, group) ENUMERATE_SIMD_(::Arcane::Face, name, group)

/*!
 * \ingroup ArcaneSimd
 * \brief SIMD enumerator over a group or list of cells.
 */
#define ENUMERATE_SIMD_CELL(name, group) ENUMERATE_SIMD_(::Arcane::Cell, name, group)

/*!
 * \ingroup ArcaneSimd
 * \brief SIMD enumerator over a group or list of particles.
 */
#define ENUMERATE_SIMD_PARTICLE(name, group) ENUMERATE_SIMD_(::Arcane::Particle, name, group)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
