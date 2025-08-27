// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdItem.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Types des entités et des énumérateurs des entités pour la vectorisation.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SIMDITEM_H
#define ARCANE_CORE_SIMDITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Simd.h"

// La macro ARCANE_SIMD_BENCH n'est définie que pour le bench
// Simd (dans contribs/Simd) et permet d'éviter d'inclure la gestion des
// entités.

#ifndef ARCANE_SIMD_BENCH
#include "arcane/core/ItemEnumerator.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file SimdItem.h
 *
 * Ce fichier contient les déclarations des types pour gérer
 * la vectorisation avec les entités (Item) du maillage.
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
 * - Faire une version de SimdItem par taille de vecteur (2,4,8).
 * - Utiliser un mask si possible.
 * - aligned SimdItemBase
 * - faire une version du constructeur de SimdItemBase sans (nb_valid)
 * pour le cas ou le vecteur est complet.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Classe gérant un vecteur SIMD d'entité.
 *
 * Cette classe conserve \a N entités du maillage, \a N étant dépendant
 * de la taille des registres SIMD et est vaut SimdInfo::Int32IndexSize.
 *
 * Cette classe ne s'utilise pas directement. Il faut utiliser SimdItem ou
 * SimdItemT
 */
class ARCANE_CORE_EXPORT SimdItemBase
{
 protected:
  
  typedef ItemInternal* ItemInternalPtr;

 public:

 typedef SimdInfo::SimdInt32IndexType SimdIndexType;

 public:

 /*!
   * \brief Construit une instance.
   * \warning \a ids doit avoir l'alignement requis pour un SimdIndexType.
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use another constructor")
  SimdItemBase(const ItemInternalPtr* items, const SimdIndexType* ids)
  : m_simd_local_ids(*ids), m_shared_info(ItemInternalCompatibility::_getSharedInfo(items,1)) { }

 protected:

  SimdItemBase(ItemSharedInfo* shared_info,const SimdIndexType* ids)
  : m_simd_local_ids(*ids), m_shared_info(shared_info) { }

 public:

  //! Partie interne (pour usage interne uniquement)
  ARCANE_DEPRECATED_REASON("Y2022: Use method SimdItem::item() instead")
  ItemInternal* item(Integer si) const { return m_shared_info->m_items_internal[localId(si)]; }

  ARCANE_DEPRECATED_REASON("Y2022: Use method SimdItem::operator[]() instead")
  ItemInternal* operator[](Integer si) const { return m_shared_info->m_items_internal[localId(si)]; }

  //! Liste des numéros locaux des entités de l'instance
  const SimdIndexType& ARCANE_RESTRICT simdLocalIds() const { return m_simd_local_ids; }

  //! Liste des numéros locaux des entités de l'instance
  const Int32* ARCANE_RESTRICT localIds() const { return (const Int32*)&m_simd_local_ids; }

  //! Numéro local de l'entité d'indice \a index.
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

  // TEMPORAIRE pour éviter le deprecated
  SimdItemDirectBase(Int32 base_local_id,Integer nb_valid,const ItemInternalPtr* items)
  : m_base_local_id(base_local_id), m_nb_valid(nb_valid),
    m_shared_info(ItemInternalCompatibility::_getSharedInfo(items,nb_valid)) { }

 public:

  //! Nombre d'entités valides de l'instance.
  inline Integer nbValid() const { return m_nb_valid; }

  //! Liste des numéros locaux des entités de l'instance
  inline Int32 baseLocalId() const { return m_base_local_id; }

 protected:

  Int32 m_base_local_id;
  Integer m_nb_valid;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Index vectoriel avec indirection pour un type d'entité.
 * TODO: stocker les index dans un registre vectoriel pour pouvoir
 * faire le gather rapidement. Pour cela, faire l'equivalent de AVXSimdReal
 * pour les Int32.
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
  //! Liste des numéros locaux des entités de l'instance
  const SimdIndexType& ARCANE_RESTRICT simdLocalIds() const { return m_local_ids; }
 private:
  const SimdIndexType& ARCANE_RESTRICT m_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Index vectoriel sans indirection pour un type d'entité
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
 * \brief Gère un vecteur d'entité \a Item.
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

  //! inline \a si-ième entité de l'instance
  inline Item item(Int32 si) const { return Item(localId(si),m_shared_info); }

  //! inline \a si-ième entité de l'instance
  inline Item operator[](Int32 si) const { return Item(localId(si),m_shared_info); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Gère un vecteur d'entité \a ItemType.
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

  //! Retourne la \a si-ième entité de l'instance
  ItemType item(Integer si) const
  {
    return ItemType(localId(si),m_shared_info);
  }

  //! Retourne la \a si-ième entité de l'instance
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
 * \brief Gère un vecteur d'entité \a ItemType.
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
 * \brief Objet permettant de positionner les valeurs d'un vecteur SIMD.
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
 * \brief Objet permettant de positionner les valeurs d'un vecteur SIMD.
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
 * \brief Classe de base des énumérateurs sur les entités vectortielles (SimdItem).
 */
class ARCANE_CORE_EXPORT SimdItemEnumeratorBase
: public SimdEnumeratorBase
{
 protected:
  
  typedef ItemInternal* ItemInternalPtr;
  
 public:

  typedef SimdInfo::SimdInt32IndexType SimdIndexType;

 public:

  // TODO: Gérer les m_local_id_offset pour cette classe

  // TODO: Fin 2024, rendre certains constructeurs internes à Arcane et rendre
  // obsolètes les autres.
  // Faire de même avec les classes dérivées

  SimdItemEnumeratorBase() = default;

  // TODO: Rendre interne à Arcane
  SimdItemEnumeratorBase(const ItemInternalVectorView& view)
  : SimdEnumeratorBase(view.localIds()), m_shared_info(view.m_shared_info) {}
  // TODO: Rendre interne à Arcane
  SimdItemEnumeratorBase(const ItemEnumerator& rhs)
  : SimdEnumeratorBase(rhs.m_view.m_local_ids,rhs.count()), m_shared_info(rhs.m_item.m_shared_info) {}

  // TODO: rendre obsolète
  SimdItemEnumeratorBase(const ItemInternalPtr* items,const Int32* local_ids,Integer n)
  : SimdEnumeratorBase(local_ids,n), m_shared_info(ItemInternalCompatibility::_getSharedInfo(items,n)) { }
  // TODO: rendre obsolète
  SimdItemEnumeratorBase(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids)
  : SimdEnumeratorBase(local_ids), m_shared_info(ItemInternalCompatibility::_getSharedInfo(items.data(),local_ids.size())) { }

 public:

  // TODO: rendre obsolète
  //! Liste des entités
  const ItemInternalPtr* unguardedItems() const { return m_shared_info->m_items_internal.data(); }

 protected:

  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur sur une liste d'entités.
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

  // TODO: rendre obsolète
  SimdItemEnumeratorT(const ItemInternalPtr* items,const Int32* local_ids,Integer n)
  : SimdItemEnumeratorBase(items,local_ids,n){}
  // TODO: rendre obsolète
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
 * \brief Vecteur SIMD de \a Node.
 */
typedef SimdItemT<Node> SimdNode;
/*!
 * \ingroup ArcaneSimd
 * \brief Vecteur SIMD de \a Edge.
 */
typedef SimdItemT<Edge> SimdEdge;
/*!
 * \ingroup ArcaneSimd
 * \brief Vecteur SIMD de \a Face.
 */
typedef SimdItemT<Face> SimdFace;
/*!
 * \ingroup ArcaneSimd
 * \brief Vecteur SIMD de \a Cell.
 */
typedef SimdItemT<Cell> SimdCell;
/*!
 * \ingroup ArcaneSimd
 * \brief Vecteur SIMD de \a Particle.
 */
typedef SimdItemT<Particle> SimdParticle;
#else
/*!
 * \ingroup ArcaneSimd
 * \brief Vecteur SIMD de \a Cell.
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
  // Créé un itérateur à partir d'un ItemVectorView. Il faut que ce dernier ait un padding
  // de la taille du vecteur.
  static SimdItemEnumeratorT<ItemType> getSimdEnumerator(const ItemVectorViewT<ItemType>& g)
  {
    return g.enumerator();
  }

  // Pour compatibilité avec l'existant
  // Si on est ici cela signifie que le type 'T' n'est pas un type Arcane.
  // Il faudrait à terme interdire cet appel (par exemple fin 2025)
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

// TODO: A supprimer. Utiliser ENUMERATE_SIMD_ à la place
#define ENUMERATE_SIMD_GENERIC(type, iname, view) \
  ENUMERATE_SIMD_(type,iname,view)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de noeuds.
 */
#define ENUMERATE_SIMD_NODE(name, group) ENUMERATE_SIMD_(::Arcane::Node, name, group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste d'arêtes.
 */
#define ENUMERATE_SIMD_EDGE(name, group) ENUMERATE_SIMD_(::Arcane::Edge, name, group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de faces.
 */
#define ENUMERATE_SIMD_FACE(name, group) ENUMERATE_SIMD_(::Arcane::Face, name, group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de mailles.
 */
#define ENUMERATE_SIMD_CELL(name, group) ENUMERATE_SIMD_(::Arcane::Cell, name, group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de particles.
 */
#define ENUMERATE_SIMD_PARTICLE(name, group) ENUMERATE_SIMD_(::Arcane::Particle, name, group)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
