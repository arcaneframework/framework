// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdItem.h                                                  (C) 2000-2018 */
/*                                                                           */
/* Types des entités et des énumérateurs des entités pour la vectorisation.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SIMDITEM_H
#define ARCANE_SIMDITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Simd.h"

// La macro ARCANE_SIMD_BENCH n'est définie que pour le bench
// Simd (dans contribs/Simd) et permet d'éviter d'inclure la gestion des
// entités.

#ifndef ARCANE_SIMD_BENCH
#include "arcane/ItemEnumerator.h"
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

ARCANE_BEGIN_NAMESPACE

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
 * Il est possible de récupérer la \a i-ème entité du vecteur via
 * l'opérateur operator[]() où la méthode item().
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
  SimdItemBase(const ItemInternalPtr* items,
               const SimdIndexType* ids)
  : m_simd_local_ids(*ids), m_items(items) { }

  //! Partie interne (pour usage interne uniquement)
  ItemInternal* item(Integer si) const { return m_items[localId(si)]; }

  ItemInternal* operator[](Integer si) const { return m_items[localId(si)]; }

  //! Liste des numéros locaux des entités de l'instance
  const SimdIndexType& ARCANE_RESTRICT simdLocalIds() const { return m_simd_local_ids; }

  //! Liste des numéros locaux des entités de l'instance
  const Int32* ARCANE_RESTRICT localIds() const { return (const Int32*)&m_simd_local_ids; }

  //! Numéro local de l'entité d'indice \a index.
  Int32 localId(Int32 index) const { return m_simd_local_ids[index]; }

 protected:

  SimdIndexType m_simd_local_ids;
  const ItemInternalPtr* m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimdItemDirectBase
{
 protected:

  typedef ItemInternal* ItemInternalPtr;
 public:
  SimdItemDirectBase(const ItemInternalPtr* items,Int32 base_local_id,Integer nb_valid)
  : m_base_local_id(base_local_id), m_nb_valid(nb_valid), m_items(items)
  {
  }

  //! Nombre d'entités valides de l'instance.
  inline Integer nbValid() const { return m_nb_valid; }

  //! Liste des numéros locaux des entités de l'instance
  inline Int32 baseLocalId() const { return m_base_local_id; }

 protected:
  Int32 m_base_local_id;
  Integer m_nb_valid;
  const ItemInternalPtr* m_items;
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

  SimdItem(const ItemInternalPtr* items,const SimdInfo::SimdInt32IndexType* ids)
  : SimdItemBase(items,ids) { }

 public:

  //! inline \a si-ième entité de l'instance
  inline Item item(Integer si) const { return m_items[localId(si)]; }

  //! inline \a si-ième entité de l'instance
  inline Item operator[](Integer si) const { return m_items[localId(si)]; }
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
 protected:
  
  typedef ItemInternal* ItemInternalPtr;

 public:
  SimdItemT(const ItemInternalPtr* items,const SimdInfo::SimdInt32IndexType* ids)
  : SimdItem(items,ids) { }

  //! Retourne la \a si-ième entité de l'instance
  ItemType item(Integer si) const
  {
    return ItemType(this->m_items[this->localId(si)]);
  }

  //! Retourne la \a si-ième entité de l'instance
  ItemType operator[](Integer si) const
  {
    return ItemType(this->m_items[this->localId(si)]);
  }

  operator SimdItemIndexT<ItemType>()
  {
    return SimdItemIndexT<ItemType>(this->simdLocalIds());
  }

private:
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
 protected:

  typedef ItemInternal* ItemInternalPtr;

 public:
  SimdItemDirectT(const ItemInternalPtr* items,Int32 base_local_id,Integer nb_valid)
  : SimdItemDirectBase(items,base_local_id,nb_valid)
  {
  }

  operator SimdItemDirectIndexT<ItemType>()
  {
    return SimdItemDirectIndexT<ItemType>(this->m_base_local_id);
  }

private:
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

  SimdItemEnumeratorBase()
  : SimdEnumeratorBase() {}
  SimdItemEnumeratorBase(const ItemInternalPtr* items,const Int32* local_ids,Integer n)
  : SimdEnumeratorBase(local_ids,n), m_items(items) {}
  SimdItemEnumeratorBase(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids)
  : SimdEnumeratorBase(local_ids), m_items(items.data()) {}
  SimdItemEnumeratorBase(const ItemInternalVectorView& view)
  : SimdEnumeratorBase(view.localIds()), m_items(view.items().data()) {}
  SimdItemEnumeratorBase(const ItemEnumerator& rhs)
  : SimdEnumeratorBase(rhs.unguardedLocalIds(),rhs.count()), m_items(rhs.unguardedItems()) {}

 public:

  //! Liste des entités
  const ItemInternalPtr* unguardedItems() const { return m_items; }

 protected:

  const ItemInternalPtr* m_items;
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
  SimdItemEnumeratorT(const ItemInternalPtr* items,const Int32* local_ids,Integer n)
  : SimdItemEnumeratorBase(items,local_ids,n){}
  SimdItemEnumeratorT(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids)
  : SimdItemEnumeratorBase(items,local_ids) {}
  SimdItemEnumeratorT(const ItemEnumerator& rhs)
  : SimdItemEnumeratorBase(rhs){}
  SimdItemEnumeratorT(const ItemVectorViewT<ItemType>& rhs)
  : SimdItemEnumeratorBase(rhs) {}
  
 public:

  SimdItemType operator*() const
  {
    return SimdItemType(m_items,_currentSimdIndex());
  }

  SimdItemDirectT<ItemType> direct() const
  {
    return SimdItemDirectT<ItemType>(m_items,m_index,nbValid());
  }

  operator SimdItemIndexT<ItemType>()
  {
    return SimdItemIndexT<ItemType>(_currentSimdIndex());
  }

#ifndef ARCANE_SIMD_BENCH
  inline ItemEnumeratorT<ItemType> enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_items,m_local_ids+m_index,nbValid(),0);
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
/*!
 * \ingroup ArcaneSimd
 * Vecteur SIMD de \a DualNode.
 */
typedef SimdItemT<DualNode> SimdDualNode;
/*!
 * \ingroup ArcaneSimd
 * \brief Vecteur SIMD de \a Link.
 */
typedef SimdItemT<Link> SimdLink;
#else
/*!
 * \ingroup ArcaneSimd
 * \brief Vecteur SIMD de \a Cell.
 */
typedef SimdItemT<Cell> SimdCell;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ENUMERATE_SIMD_GENERIC(type,iname,view)                         \
  for( A_TRACE_ITEM_ENUMERATOR(SimdItemEnumeratorT< type >) iname((view).enumerator() A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname )

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de noeuds.
 */
#define ENUMERATE_SIMD_NODE(name,group) ENUMERATE_SIMD_GENERIC(::Arcane::Node,name,group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste d'arêtes.
 */
#define ENUMERATE_SIMD_EDGE(name,group) ENUMERATE_SIMD_GENERIC(::Arcane::Edge,name,group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de faces.
 */
#define ENUMERATE_SIMD_FACE(name,group) ENUMERATE_SIMD_GENERIC(::Arcane::Face,name,group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de mailles.
 */
#define ENUMERATE_SIMD_CELL(name,group) ENUMERATE_SIMD_GENERIC(::Arcane::Cell,name,group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de particles.
 */
#define ENUMERATE_SIMD_PARTICLE(name,group) ENUMERATE_SIMD_GENERIC(::Arcane::Particle,name,group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de DualNode.
 */
#define ENUMERATE_SIMD_DUALNODE(name,group) ENUMERATE_SIMD_GENERIC(::Arcane::DualNode,name,group)

/*!
 * \ingroup ArcaneSimd
 * \brief Enumérateur SIMD sur un groupe ou liste de Link.
 */
#define ENUMERATE_SIMD_LINK(name,group) ENUMERATE_SIMD_GENERIC(::Arcane::Link,name,group)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
