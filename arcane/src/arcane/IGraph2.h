/*---------------------------------------------------------------------------*/
/* IGraph2.h                                                    (C) 2011-2011 */
/*                                                                           */
/* Interface d'un graphe d'un maillage    .                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IGRAPH2_H
#define ARCANE_IGRAPH2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"
#include "arcane/IItemConnectivity.h"
#include "arcane/IndexedItemConnectivityView.h"
#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
//#include <typeinfo>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Nombre de type d'entit�s duales
static const Integer NB_DUAL_ITEM_TYPE = 5;

extern "C++" ARCANE_CORE_EXPORT eItemKind
dualItemKind(Integer type);

class IGraphModifier2;
class IGraph2 ;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Outillage de connectivit� d'un graphe
 */

class ARCANE_CORE_EXPORT GraphConnectivity
{
 public :
  GraphConnectivity(IItemConnectivity* link_connectivity,
      UniqueArray<ItemConnectivity> const& dual_item_connectivities,
      ItemScalarProperty<Integer> const& dual_node_to_connectivity)
  : m_link_connectivity{ link_connectivity }
  , m_connectivities(dual_item_connectivities)
  , m_dual_node_to_connectivity_index(dual_node_to_connectivity)
  {

  }

  inline Item dualItem(const DoF& dualNode) const {
    return m_connectivities[m_dual_node_to_connectivity_index[dualNode]](dualNode);
  }

  inline DoFVectorView dualNodes(const DoF& link) {
    return m_link_connectivity.connectedItems(link);
  }

  ConnectivityItemVector dualNodesConnectivityVector() const {
    return m_link_connectivity;
  }

 private :
  ConnectivityItemVector m_link_connectivity;
  UniqueArray<ItemConnectivity> const & m_connectivities;
  ItemScalarProperty<Integer> const& m_dual_node_to_connectivity_index;

};

class ARCANE_CORE_EXPORT GraphIncrementalConnectivity
{
 public :
  GraphIncrementalConnectivity(IItemFamily const* dualnode_family,
                               IItemFamily const* link_family,
                               Arcane::mesh::IncrementalItemConnectivity* link_connectivity,
                               UniqueArray<Arcane::mesh::IncrementalItemConnectivity*> const& dualitem_connectivities,
                               ItemScalarProperty<Integer> const& dualnode_to_connectivity)
  : m_dualnode_family(dualnode_family)
  , m_link_family(link_family)
  , m_link_connectivity(link_connectivity)
  , m_link_connectivity_accessor(link_connectivity->connectivityAccessor())
  , m_dualitem_connectivities(dualitem_connectivities)
  , m_dualnode_to_connectivity_index(dualnode_to_connectivity)
  {
    m_dualitem_connectivity_accessors.resize(m_dualitem_connectivities.size()) ;
    for(Integer i=0;i<m_dualitem_connectivities.size();++i)
    {
      if(m_dualitem_connectivities[i])
      {
        m_dualitem_connectivity_accessors[i] = m_dualitem_connectivities[i]->connectivityAccessor() ;
      }
    }
  }

  GraphIncrementalConnectivity(GraphIncrementalConnectivity const& rhs)
  : m_link_connectivity(rhs.m_link_connectivity)
  , m_link_connectivity_accessor(m_link_connectivity->connectivityAccessor())
  , m_dualitem_connectivities(rhs.m_dualitem_connectivities)
  , m_dualnode_to_connectivity_index(rhs.m_dualnode_to_connectivity_index)
  {
    m_dualitem_connectivity_accessors.resize(m_dualitem_connectivities.size()) ;
    for(Integer i=0;i<m_dualitem_connectivities.size();++i)
    {
      if(m_dualitem_connectivities[i])
      {
        m_dualitem_connectivity_accessors[i] = m_dualitem_connectivities[i]->connectivityAccessor() ;
      }
    }
  }

  inline Item dualItem(const DoF& dualNode) const
  {
    return m_dualitem_connectivity_accessors[m_dualnode_to_connectivity_index[dualNode]](ItemLocalId(dualNode))[0];
  }

  inline DoFVectorView dualNodes(const DoF& link) const
  {
    return m_link_connectivity_accessor(ItemLocalId(link));
  }

 private :
  IItemFamily const*                         m_dualnode_family   = nullptr ;
  IItemFamily const*                         m_link_family       = nullptr ;
  Arcane::mesh::IncrementalItemConnectivity* m_link_connectivity = nullptr;
  IndexedItemConnectivityAccessor                                 m_link_connectivity_accessor ;
  UniqueArray<Arcane::mesh::IncrementalItemConnectivity*> const & m_dualitem_connectivities;
  UniqueArray<IndexedItemConnectivityAccessor>                    m_dualitem_connectivity_accessors ;
  ItemScalarProperty<Integer> const& m_dualnode_to_connectivity_index;

};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un graphe du maillage
 */
class ARCANE_CORE_EXPORT IGraph2
{
public:

  virtual ~IGraph2() {} //<! Lib�re les ressources

public:

  virtual IGraphModifier2* modifier() =0;

//#define GRAPH_USE LEGACY_CONNECTIVITY
#define GRAPH_USE_INCREMENTAL_CONNECTIVITY
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  typedef GraphConnectivity GraphConnectivityType ;
  GraphConnectivity const* connectivity() const {
    return legacyConnectivity() ;
  }
  virtual GraphConnectivity const* legacyConnectivity() const =0;
#endif
#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
  typedef GraphIncrementalConnectivity GraphConnectivityType ;
  GraphIncrementalConnectivity const* connectivity() const {
    return incrementalConnectivity() ;
  }
  virtual GraphIncrementalConnectivity const* incrementalConnectivity() const =0;
#endif

public:

  //! Nombre de noeuds duaux du graphe
  virtual Integer nbDualNode() const =0;

  //! Nombre de liaisons du graphe
  virtual Integer nbLink() const =0;

public:

  //! Retourne la famille des noeuds duaux
  virtual const IItemFamily* dualNodeFamily() const = 0;
  virtual IItemFamily* dualNodeFamily() = 0;

  //! Retourne la famille des liaisons
  virtual const IItemFamily* linkFamily() const = 0;
  virtual IItemFamily* linkFamily() = 0;

  virtual void printDualNodes() const = 0;
  virtual void printLinks() const = 0;

};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
