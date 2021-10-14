// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DirectedAcyclicGraphT.h                                     (C) 2000-2017 */
/*                                                                           */
/* Implementation of a directed acyclic graph                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DIRECTEDACYCLICGRAPHT_H_ 
#define ARCANE_DIRECTEDACYCLICGRAPHT_H_ 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <iterator>

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/GraphBaseT.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Math.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! Template class for DirectedAcyclicGraph.
 *  VertexType must implement a less comparison operator.
 *
 */

template <class VertexType, class EdgeType>
class DirectedAcyclicGraphT
: public GraphBaseT<VertexType,EdgeType>
{
  typedef VertexType& VertexRef;
  typedef EdgeType& EdgeRef;
public:

  /** Constructeur de la classe */
  DirectedAcyclicGraphT(ITraceMng* trace_mng)
  : GraphBaseT<VertexType,EdgeType>(trace_mng)
  , m_compute_vertex_levels(true){}

  /** Destructeur de la classe */
  virtual ~DirectedAcyclicGraphT() {}

public:

  template <class ContainerT>
   class SortedElementSet
   {
   public:
    template <class T>
    class ReverseOrderSet
    {
    public:
      ReverseOrderSet(const T& elements) : m_elements(elements){}

      virtual ~ReverseOrderSet() {}

      typedef typename std::reverse_iterator<typename T::iterator> iterator;
      typedef typename std::reverse_iterator<typename T::const_iterator> const_iterator;

      iterator begin() {return iterator(m_elements.end()); }
      const_iterator begin() const {return iterator(m_elements.end()); ;}

      iterator end() {return iterator(m_elements.begin());}
      const_iterator end() const {return iterator(m_elements.begin());}

      Integer size() {return m_elements.size();}

    private:
      T m_elements;

    };
   public:
     SortedElementSet(const ContainerT& elements) : m_elements(elements) {}

     SortedElementSet(ContainerT&& elements) : m_elements(elements) {}

     SortedElementSet() {}

     virtual ~SortedElementSet(){}

     typedef typename ContainerT::iterator iterator;
     typedef typename ContainerT::const_iterator const_iterator;

     iterator begin() {return m_elements.begin();}
     const_iterator begin() const {return m_elements.begin();}

     iterator end() {return m_elements.end();}
     const_iterator end() const {return m_elements.end();}

     Integer size() {return m_elements.size();}

     ReverseOrderSet<ContainerT> reverseOrder() {return ReverseOrderSet<ContainerT>(m_elements);}

   private:
     ContainerT m_elements;
   };

  typedef GraphBaseT<VertexType,EdgeType> Base;
  typedef SortedElementSet<typename Base::EdgeTypeRefArray> SortedEdgeSet;
  typedef SortedElementSet<typename Base::VertexTypeRefArray> SortedVertexSet;
  typedef std::map<typename Base::VertexTypeConstRef,Integer> VertexLevelMap;
  typedef std::map<typename Base::EdgeTypeConstRef,Integer> EdgeLevelMap;
  typedef std::set<std::pair<VertexType,Integer>, std::function<bool (std::pair<VertexType,Integer>,std::pair<VertexType,Integer>)>> VertexLevelSet;

  void addEdge(const VertexType& source_vertex, const VertexType& target_vertex, const EdgeType& source_to_target_edge)
    {
      Base::addEdge(source_vertex,target_vertex,source_to_target_edge);
      m_compute_vertex_levels = true;
    }

    void addEdge(VertexType&& source_vertex, VertexType&& target_vertex, EdgeType&& source_to_target_edge)
    {
      Base::addEdge(source_vertex,target_vertex,source_to_target_edge);
      m_compute_vertex_levels = true;
    }

  SortedVertexSet topologicalSort()
  {
    if (m_compute_vertex_levels) _computeVertexLevels();
    typename Base::VertexTypeRefArray sorted_vertices;
    for (auto& vertex : this->m_vertices) {sorted_vertices.add(std::ref(vertex));}
    std::sort(sorted_vertices.begin(),sorted_vertices.end(),[&](const VertexType& a, const VertexType& b){
      return m_vertex_level_map[a] < m_vertex_level_map[b];});
    return SortedVertexSet(std::move(sorted_vertices));
  }

  SortedEdgeSet spanningTree()
  {
    if (m_compute_vertex_levels) _computeVertexLevels();
    typename Base::EdgeTypeRefArray spaning_tree_edges;
    for (auto& edge : this->m_edges)
      {
        Integer level = _computeEdgeLevel(edge);
        m_edge_level_map[edge] = level;
        if (level < 0) continue; // edge must not be accounted for in spanning tree
      spaning_tree_edges.add(std::ref(edge));
      }
    std::sort(spaning_tree_edges.begin(),spaning_tree_edges.end(), [&](const EdgeType& a, const EdgeType& b){
        return m_edge_level_map[a] < m_edge_level_map[b];});
    return SortedEdgeSet(spaning_tree_edges);
  }

  void print()
  {
    if (m_compute_vertex_levels) _computeVertexLevels();
    this->m_trace_mng->info() << "--- Directed Graph ---";
    for (auto vertex_entry: this->m_adjacency_list)
      {
        _printGraphEntry(vertex_entry);
      }

    std::ostringstream oss;
    this->m_trace_mng->info() << oss.str();

    for (auto vertex_level_set_entry : m_vertex_level_map)
      {
        this->m_trace_mng->info() << "-- Graph has vertex " << vertex_level_set_entry.first << " with level " << vertex_level_set_entry.second;
      }
  }

  /*! La detection de cycle se fait avec un pattern lazy qui n'est lancé que lors de l'appel à topologicalSort() et print().
   *  Cette méthode permet de savoir si le graphe contient un cycle (ce qui produirait l'échec de topologicalSort())
   */
    bool hasCycle()
    {
      bool has_cycle = false;
      try {
          _computeVertexLevels();
      } catch (const FatalErrorException& e) {
          has_cycle = true;
      }
      return has_cycle;
    }

private:
  std::set<typename Base::VertexTypeConstRef> m_colored_vertices;
  VertexLevelMap m_vertex_level_map;
  EdgeLevelMap m_edge_level_map;
  bool m_compute_vertex_levels;

private:

  void _computeVertexLevels()
  {
    // Current algo cannot update vertex level ; need to clear the map.
    m_vertex_level_map.clear();
    // compute vertex level
    for (auto vertex_entry : this->m_adjacency_list)
      {
        _computeVertexLevel(vertex_entry.first,0);
      }
    m_compute_vertex_levels = false;
  }

  void _computeVertexLevel(const VertexType& vertex, Integer level)
  {
    // Check for cycles
    if (! m_colored_vertices.insert(std::cref(vertex)).second) throw FatalErrorException("Cycle in graph. Exiting");

    // Try to insert vertex at the given level
    bool update_children = true;
    auto vertex_level_set_entry = m_vertex_level_map.insert(std::make_pair(std::cref(vertex),level)); // use emplace when available (gcc >= 4.8.0)
    if (!vertex_level_set_entry.second) // vertex already present
      {
        if (vertex_level_set_entry.first->second < level) vertex_level_set_entry.first->second = level;
        else update_children = false;
      }
    if (update_children)
      {
        auto vertex_adjacency_list = Base::m_adjacency_list.find(vertex);
        if (vertex_adjacency_list != Base::m_adjacency_list.end())
          {
            ++level;
            for (auto child_vertex : vertex_adjacency_list->second.first)
              {
                _computeVertexLevel(child_vertex,level);
              }
          }
      }
    // Remove vertex from cycle detection
    m_colored_vertices.erase(vertex);
  }

  Integer _computeEdgeLevel(const EdgeType& edge)
  {
    auto edge_vertices_iterator = this->m_edge_to_vertex_map.find(edge);
    if (edge_vertices_iterator == this->m_edge_to_vertex_map.end()) throw FatalErrorException("Not existing edge.");
    typename Base::VertexPair edge_vertices = edge_vertices_iterator->second;
    // edge level is set to source vertex level
    Integer edge_level = m_vertex_level_map[std::cref(edge_vertices.first)];
    // if the edge crosses graph levels (ie vertices level difference > 1), we put edge_level =-1, since we don't want to keep it in the spanning tree
    if (math::abs(edge_level - m_vertex_level_map[std::cref(edge_vertices.second)]) >1) edge_level = -1;
    return edge_level;
  }

  void _printGraphEntry(const typename Base::AdjacencyListType::value_type& vertex_entry)
  {
    this->m_trace_mng->info() << "-- Vertex " << vertex_entry.first << " depends on ";
    for (auto connected_vertex : vertex_entry.second.first )
      {
        this->m_trace_mng->info() << "  - " << connected_vertex;
      }
  }

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* DIRECTEDACYCLICGRAPHT_H_ */
