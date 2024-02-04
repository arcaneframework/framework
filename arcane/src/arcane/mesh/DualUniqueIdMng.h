// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCANE_MESH_DUALUNIQUEIDMNG_H
#define ARCANE_MESH_DUALUNIQUEIDMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include <utility>
#include "arcane/utils/String.h"
#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Stratégie de numérotation des UniqueId des item (DualNode, Link) d'un graphe
 *
 * DualNode :
 * --------
 *
 * L'UniqueId est crée à partir de celui de l'item
 * Pour le moment, on suppose que l'UniqueId de l'item est codé sur 30 bits,
 * ce qui signifie qu'on autorise 2^30 items (environ 1 milliard)
 *
 * DualNode avec DualItem de type Node : 
 *       [ 29 bit (Node UniqueId)   | 
 *         32 bit (0)               |
 *          2 bit (Code Node 0 0)   |
 *          1 bit (Signe positif 0 )]
 *       = 64 bit
 *         
 * DualNode avec DualItem de type Face : 
 *       [ 29 bit (Face UniqueId)   | 
 *         32 bit (0)               |
 *          2 bit (Code Face 0 1)   |
 *          1 bit (Signe positif 0 )] 
 *       = 64 bit
 *
 * DualNode avec DualItem de type Cell : 
 *       [ 29 bit (Cell UniqueId)   | 
 *         32 bit (0)               | 
 *          2 bit (Code Cell 1 0)   | 
 *          1 bit (Signe positif 0 )]
 *       = 64 bit
 * 
 * DualNode avec DualItem de type Edge :
 *       [ 29 bit (Edge UniqueId)   |
 *         32 bit (0)               |
 *          2 bit (Code Edge 1 1)   |
 *          1 bit (Signe positif 0 )]
 *       = 64 bit
 *
 * On donne la possibilité de créer plusieurs DualNodes par item. Dans ce cas, les DualNodes sont
 * différentier par leur rang (sur 4 bits soit 2^4=16 dualnodes par items au maximum)
 * Dans ce cas d'utilisation on autorise seulement 2^25 items (=33 554 432 items)
 *
 * DualNode avec DualItem de type Node|Face|Cell|Edge :
 *       [ 25 bit (Item UniqueId)   |
 *       [  4 bit (DualItem rank)   |
 *         32 bit (0)               |
 *          2 bit (Code Edge 1 1)   |
 *          1 bit (Signe positif 0 )]
 *       = 64 bit
 *
 * Link :
 * ----
 *
 * On concatène les UniqueIds des items que joint le lien Link
 *
 * Link liant Item_1 et Item_2 = 
 *       [ 29 bit (Item_1 UniqueId) | 
 *         29 bit (Item_2 UniqueId) |
 *          1 bit (0)               |
 *          2 bit (Code Item_1)     | 
 *          2 bit (Code Item_2)     |
 *          1 bit (Signe positif 0 )]
 *       = 64 bit
 *
 * Si plusieurs dualnodes par item, on précise les rangs des dualnodes des l'items liés.
 *
 * Link liant DualItem_1 rang_1 et DualItem_2 rang_2 =
 *       [ 25 bit (Item_1 UniqueId)   |
 *       [  4 bit (DualItem_1 rank)   |
 *       [ 25 bit (Item_2 UniqueId)   |
 *       [  4 bit (DualItem_2 rank)   |
 *          1 bit (0)               |
 *          2 bit (Code Item_1)     |
 *          2 bit (Code Item_2)     |
 *          1 bit (Signe positif 0 )]
 *       = 64 bit
 *
 *
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/Item.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT DualUniqueIdMng
  : public TraceAccessor
{
private:

  using TraceAccessor::info;

public:

  static const Int64 node_code = 0;
  static const Int64 face_code = Int64(1) << 62;
  static const Int64 cell_code = Int64(1) << 61;
  static const Int64 edge_code = (Int64(1) << 61) + (Int64(1) << 62);
  static const Int64 particle_code = (Int64(1) << 61) + (Int64(1) << 62);

  bool m_use_dual_particle = true ;

  DualUniqueIdMng(ITraceMng * trace_mng,bool use_dual_particle=true)
  : TraceAccessor(trace_mng)
  , m_use_dual_particle(use_dual_particle)
  {}

  ~DualUniqueIdMng() {}

public:
  inline eItemKind codeToItemKind(Int64 code) ;
  inline eItemKind uidToDualItemKind(Int64 unique_id) ;
  inline Int64 uniqueIdOf(eItemKind item_kind, Int64 item_uid);

  template<typename ItemT>
  inline static Int64 uniqueIdOf(const ItemT& item);

  template<typename ItemT>
  inline Int64 debugUniqueIdOf(const ItemT& item);

  inline std::tuple<eItemKind,Int64> uniqueIdOfDualItem(const DoF& item);

  template<typename ItemT>
  inline static Int64 uniqueIdOf(const ItemT& item, const Integer rank);

  template<typename ItemT_1, typename ItemT_2>
  inline static Int64 uniqueIdOf(const ItemT_1& item_1, const ItemT_2& item_2);

  inline std::pair< std::tuple<eItemKind,Int64>,std::tuple<eItemKind,Int64> > uniqueIdOfPairOfDualItems(const DoF& item);

  template<typename ItemT_1, typename ItemT_2>
  inline static Int64 uniqueIdOf(const ItemT_1& item_1, const Integer item_1_rank,
      const ItemT_2& item_2, const Integer item_2_rank);

  inline static Integer rankOf(const DoF& );

  inline void info(const DoF& node, const Item& dual_item) const;
  inline void info(const DoF& link, const DoF& dual_node0, const DoF& dual_node1, const Item& dual_item0, const Item& dual_item1) const;

  inline Int64 debugDualItemUniqueId(DoF& node) const;

private:

  template<typename ItemT, typename Type>
  struct traits_item_code;

  template<Integer Nbit,typename Type>
  inline static bool _onlyFirstBitUsed(const Type id);

  inline bool _checkDualNode(const DoF& node, const Item& dual_item) const;
  inline bool _checkLink    (const DoF& link, const Item& dual_item0, const Item& dual_item1) const;

  inline Int64 _extractFirstCode (const Int64 id) const;
  inline Int64 _extractSecondCode(const Int64 id) const;
  inline Int64 _extractFirstId (const Int64 id) const;
  inline Int64 _extractSecondId(const Int64 id) const;

  inline bool _codeIsValid(const Item& item, const Int64 code) const;
  inline bool   _idIsValid(const Item& item, const Int64 id  ) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Code des DualNode en fonction des Item
template<>
struct ARCANE_MESH_EXPORT DualUniqueIdMng::traits_item_code<Node,Int64>
{
  static const Int64 code = 0;
};

template<>
struct ARCANE_MESH_EXPORT DualUniqueIdMng::traits_item_code<Face,Int64>
{
  static const Int64 code = Int64(1) << 62;
};

template<>
struct ARCANE_MESH_EXPORT DualUniqueIdMng::traits_item_code<Cell,Int64>
{
  static const Int64 code = Int64(1) << 61;
}
;
template<>
struct ARCANE_MESH_EXPORT DualUniqueIdMng::traits_item_code<Edge,Int64>
{
  static const Int64 code = (Int64(1) << 61) + (Int64(1) << 62);
};

template<>
struct ARCANE_MESH_EXPORT DualUniqueIdMng::traits_item_code<Particle,Int64>
{
  //! attention incompatible avec une utilisation silmutanée de dual node sur des arêtes et des particules
  static const Int64 code = (Int64(1) << 61) + (Int64(1) << 62);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Verifie que seuls les Nbit premiers bits sont utilisés
//
// Exemple:
// bool onlyFirstBitUsed<4,Integer>(Integer id) avec sizeof(Integer) = 8 bits
// la méthode renvoie vrai si l'entier id est codé sur les 4 premiers bits,
// (ie si les 4 derniers bits sont à 0)
// 
// Par cela, on cree un filtre valant 1 pour les 4 derniers bits et on
// utilise la comparaison binaire &. Si le résultat est nul, on renvoie vrai
//
template<Integer Nbit,typename Type>
inline bool
DualUniqueIdMng::
_onlyFirstBitUsed(const Type id)
{
  ARCANE_ASSERT((Nbit > 0),("Error template parameter Nbit <= 0"));

  [[maybe_unused]] const Integer nb_bit_max = (Integer)(8*sizeof(Type));

  ARCANE_ASSERT((Nbit < nb_bit_max),("Error 8*sizeof(Type) <= Nbit"));

  // filtre sur les nb_bit_max - Nbit derniers bits
  const Type Nbit_first_bits_nulls = ~ ( (1 << Nbit) - 1 );

  // Si les nb_bit_max - Nbit derniers bits sont nuls, vrai
  return (Nbit_first_bits_nulls & id) == Type(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemT>
inline Int64
DualUniqueIdMng::
uniqueIdOf(const ItemT& item)
{
  ARCANE_ASSERT((8*sizeof(Int64) == 64),("Int64 is not 64-bits"));
  ARCANE_ASSERT((_onlyFirstBitUsed<29,Int64>(item.uniqueId())),
                (String::format("Item kind={0} uid={1} : invalid uid (more than 29 bit)",
                                itemKindName(item.kind()),item.uniqueId()).localstr()));

  const Int64 unique_id = item.uniqueId();

  return unique_id | traits_item_code<ItemT,Int64>::code;
}


template<typename ItemT>
inline Int64
DualUniqueIdMng::
debugUniqueIdOf(const ItemT& item)
{
  ARCANE_ASSERT((8*sizeof(Int64) == 64),("Int64 is not 64-bits"));
  ARCANE_ASSERT((_onlyFirstBitUsed<29,Int64>(item.uniqueId())),
                (String::format("Item kind={0} uid={1} : invalid uid (more than 29 bit)",
                                itemKindName(item.kind()),item.uniqueId()).localstr()));

  const Int64 unique_id = item.uniqueId();

  return unique_id | traits_item_code<ItemT,Int64>::code;
}

inline eItemKind
DualUniqueIdMng::
uidToDualItemKind(Int64 unique_id)
{
  Int64 code = _extractSecondCode(unique_id) ;

  if(code==face_code)
      return IK_Face ;
  if(code==node_code)
      return  IK_Node ;
  if(code==cell_code)
      return  IK_Cell ;
  if(m_use_dual_particle && code==particle_code)
    return  IK_Particle ;
  else if(code==edge_code)
      return  IK_Edge ;
  return IK_Unknown ;
}


inline eItemKind
DualUniqueIdMng::
codeToItemKind(Int64 code)
{
  if(code==face_code)
      return IK_Face ;
  if(code==node_code)
      return  IK_Node ;
  if(code==cell_code)
      return  IK_Cell ;
  if(m_use_dual_particle && code==particle_code)
    return  IK_Particle ;
  else if(code==edge_code)
    return  IK_Edge ;

  return IK_Unknown ;
}

inline Int64
DualUniqueIdMng::
uniqueIdOf(eItemKind item_kind, Int64 item_uid)
{
  ARCANE_ASSERT((8*sizeof(Int64) == 64),("Int64 is not 64-bits"));
  ARCANE_ASSERT((_onlyFirstBitUsed<29,Int64>(item_uid)),
                (String::format("Item kind={0} uid={1} : invalid uid (more than 29 bit)",
                                itemKindName(item_kind),item_uid).localstr()));

  switch(item_kind)
  {
    case IK_Node :
      return item_uid | node_code ;
    case IK_Face :
      return item_uid | face_code ;
    case IK_Cell :
      return item_uid | cell_code ;
    case IK_Edge :
      return item_uid | edge_code ;
    case IK_Particle :
      return item_uid | particle_code ;
    default :
      throw FatalErrorException(A_FUNCINFO,"Item not defined in graph");
  }
  return -1 ;
}

inline std::tuple<eItemKind,Int64>
DualUniqueIdMng::
uniqueIdOfDualItem(const DoF& node)
{
  ARCANE_ASSERT((8*sizeof(Int64) == 64),("Int64 is not 64-bits"));
  const Int64 node_id = node.uniqueId();
  const Int64 dual_id = _extractFirstId(node_id);
  eItemKind item_kind = uidToDualItemKind(node_id) ;

  return std::make_tuple(item_kind,dual_id) ;
}



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemT>
inline Int64
DualUniqueIdMng::
uniqueIdOf(const ItemT& item, const Integer rank)
{
  ARCANE_ASSERT((8*sizeof(Int64) == 64),("Int64 is not 64-bits"));
  ARCANE_ASSERT((_onlyFirstBitUsed<25,Int64>(item.uniqueId())),
                (String::format("Item kind={0} uid={1} : invalid uid (more than 25 bit)",
                                itemKindName(item.kind()),item.uniqueId()).localstr()));
  ARCANE_ASSERT((_onlyFirstBitUsed<4,Int64>(Int64(rank))),
                (String::format("rank={0} : invalid level (more than 4 bit)", rank).localstr()));

  const Int64 unique_id = item.uniqueId();

  return unique_id | Int64(rank) << 25 | traits_item_code<ItemT,Int64>::code;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemT_1, typename ItemT_2>
inline Int64
DualUniqueIdMng::
uniqueIdOf(const ItemT_1& item_1, const ItemT_2& item_2)
{
  ARCANE_ASSERT((8*sizeof(Int64) == 64),("Int64 is not 64-bits"));
  ARCANE_ASSERT((_onlyFirstBitUsed<29,Int64>(item_1.uniqueId())),
                (String::format("Item kind={0} uid={1} : invalid uid (more than 29 bit)",
                                itemKindName(item_1.kind()),item_1.uniqueId()).localstr()));
  ARCANE_ASSERT((_onlyFirstBitUsed<29,Int64>(item_2.uniqueId())),
                (String::format("Item kind={0} uid={1} : invalid uid (more than 29 bit)",
                                itemKindName(item_2.kind()),item_2.uniqueId()).localstr()));

  const Int64 item_1_unique_id = item_1.uniqueId();
  const Int64 item_2_unique_id = item_2.uniqueId();

  return item_1_unique_id                          | // id de l'item 1 sur 29 bits
         item_2_unique_id << 29                    | // id de l'item 2 sur 29 bits suivants
         traits_item_code<ItemT_1,Int64>::code >> 2 | // code de l'item 1 sur 2 bits suivants
         traits_item_code<ItemT_2,Int64>::code;       // code de l'item 2 sur 2 derniers bits
}


inline std::pair<std::tuple<eItemKind,Int64>,std::tuple<eItemKind,Int64> >
DualUniqueIdMng::
uniqueIdOfPairOfDualItems(const DoF& link)
{
  ARCANE_ASSERT((8*sizeof(Int64) == 64),("Int64 is not 64-bits"));

  const Int64 link_id = link.uniqueId();

  const Int64     code_1 = _extractFirstCode(link_id);
  const Int64       id_1 = _extractFirstId(link_id);
  const eItemKind kind_1 = codeToItemKind(code_1) ;

  const Int64     code_2 = _extractSecondCode(link_id);
  const Int64       id_2 = _extractSecondId(link_id);
  const eItemKind kind_2 = codeToItemKind(code_2) ;


  return std::make_pair(std::make_tuple(kind_1,id_1),std::make_tuple(kind_2,id_2)) ;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemT_1, typename ItemT_2>
inline Int64
DualUniqueIdMng::
uniqueIdOf(const ItemT_1& item_1, const Integer item_1_rank, const ItemT_2& item_2, const Integer item_2_rank)
{
  ARCANE_ASSERT((8*sizeof(Int64) == 64),("Int64 is not 64-bits"));
  ARCANE_ASSERT((_onlyFirstBitUsed<25,Int64>(item_1.uniqueId())),
                (String::format("Item kind={0} uid={1} : invalid uid (more than 25 bit)",
                                itemKindName(item_1.kind()),item_1.uniqueId()).localstr()));
  ARCANE_ASSERT((_onlyFirstBitUsed<4,Int64>(Int64(item_1_rank))),
                (String::format("rank={0} : invalid level (more than 4 bit)", item_1_rank).localstr()));
  ARCANE_ASSERT((_onlyFirstBitUsed<25,Int64>(item_2.uniqueId())),
                (String::format("Item kind={0} uid={1} : invalid uid (more than 25 bit)",
                                itemKindName(item_2.kind()),item_2.uniqueId()).localstr()));
  ARCANE_ASSERT((_onlyFirstBitUsed<4,Int64>(Int64(item_2_rank))),
                (String::format("rank={0} : invalid level (more than 4 bit)", item_2_rank).localstr()));

  const Int64 item_1_unique_id = item_1.uniqueId();
  const Int64 item_2_unique_id = item_2.uniqueId();

  return item_1_unique_id                           | // id de l'item 1 sur 25 bits
         Int64(item_1_rank) << 25                   | // rang de l'item 1 sur 4 bits
         item_2_unique_id << 29                     | // id de l'item 2 sur 25 bits suivants
         Int64(item_2_rank) << 54                   | // rang de l'item 2 sur 4 bits suivants
         traits_item_code<ItemT_1,Int64>::code >> 2 | // code de l'item 1 sur 2 bits suivants
         traits_item_code<ItemT_2,Int64>::code;       // code de l'item 2 sur 2 derniers bits
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
inline  Integer
DualUniqueIdMng::
rankOf(const DoF& node)
{
  const Int64 id = node.uniqueId();
  return Integer(( ~( (Int64(1) << 25) - 1) & id ) >> 25);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
inline bool
DualUniqueIdMng::
_codeIsValid(const Item& item, const Int64 code) const
{
  const eItemKind item_kind = item.kind();
  switch(item_kind)
  {
  case IK_Face :
    if(code != traits_item_code<Face,Int64>::code) {
      return false;
    }
    break;
  case IK_Node :
    if(code != traits_item_code<Node,Int64>::code) {
      return false;
    }
    break;
  case IK_Cell :
    if(code != traits_item_code<Cell,Int64>::code) {
      return false;
    }
    break;
  case IK_Edge :
    if(code != traits_item_code<Edge,Int64>::code) {
      return false;
    }
    break;
  case IK_Particle :
    if(code != traits_item_code<Particle,Int64>::code) {
      return false;
    }
    break;
  default:
    throw FatalErrorException(A_FUNCINFO,"Item not defined in graph");
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
DualUniqueIdMng::
_idIsValid(const Item& item, const Int64 id) const
{
  return (id != item.uniqueId()) ? false : true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Int64
DualUniqueIdMng::
_extractSecondCode(const Int64 id) const
{
  return ~ ( (Int64(1) << 61) - 1 ) & id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Int64
DualUniqueIdMng::
_extractFirstId(const Int64 id) const
{
  return ( (Int64(1) << 29) - 1 ) & id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Int64
DualUniqueIdMng::
_extractFirstCode(const Int64 id) const
{
  return ( ~( ((Int64(1) << 59) - 1) | ~((Int64(1) << 61) - 1 ) ) & id ) << 2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Int64
DualUniqueIdMng::
_extractSecondId(const Int64 id) const
{
  return ( ~( ((Int64(1) << 29) - 1) | ~((Int64(1) << 59) - 1 ) ) & id ) >> 29;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
DualUniqueIdMng::
_checkDualNode(const DoF& node, const Item& dual_item) const
{
  const Int64 node_id = node.uniqueId();

  const Int64 code = _extractSecondCode(node_id);
  const Int64 id   = _extractFirstId(node_id);


  return _codeIsValid(dual_item,code) && _idIsValid(dual_item,id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
DualUniqueIdMng::
_checkLink(const DoF& link, const Item& item_1, const Item& item_2) const
{

  const Int64 link_id = link.uniqueId();

  const Int64 code_1 = _extractFirstCode(link_id);
  const Int64   id_1 = _extractFirstId(link_id);

  const Int64 code_2 = _extractSecondCode(link_id);
  const Int64   id_2 = _extractSecondId(link_id);


  return _codeIsValid(item_1,code_1) && _idIsValid(item_1,id_1)
      && _codeIsValid(item_2,code_2) && _idIsValid(item_2,id_2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
DualUniqueIdMng::
info(const DoF& node, const Item& dual_item) const
{
  ARCANE_ASSERT((_checkDualNode(node,dual_item) == true),("Error from dual node consistence. Do you use DualUniqueIdMng to generate unique id of graph item ?"));


  info() << " -- Dual Node with unique id " << node.uniqueId()
         << " of item of kind " << dual_item.kind()
         << " and unique id " << dual_item.uniqueId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
DualUniqueIdMng::
info(const DoF& link, const DoF& dual_node0, const DoF& dual_node1,const Item& dual_item0, const Item& dual_item1) const
{
  ARCANE_ASSERT((_checkLink(link,dual_item0,dual_item1) == true),("Error from link consistence. Do you use DualUniqueIdMng to generate unique id of graph item ?"));

  info() << "- Link with unique id " << link.uniqueId() << " of :";

  info(dual_node0,dual_item0);

  info() << " and :";

  info(dual_node1,dual_item1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* MESH_UTILS_DUALUNIQUEIDMNG_H */
