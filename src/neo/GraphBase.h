//
// Created by dechaiss on 23/12/2020.
//

#ifndef NEO_GRAPHBASE_H
#define NEO_GRAPHBASE_H

#include <map>
#include <set>
#include <list>
#include <functional>
#include <algorithm>
#include <utility>
#include <vector>
#include <numeric>

#include "neo/Utils.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! Template base class for Graph.
 *  VertexType must implement a less comparison operator.
 *  To use print, VertexType must implement << operator
 *  Multiple Edges between the same Vertices are not allowed
 */

// TODO EdgeType = void (default)
// TODO add a template argument Comparator = std::less

namespace Neo {

    template<class VertexType, class EdgeType>
    class GraphBase {
    protected:

        GraphBase() = default;

        virtual ~GraphBase() = default;

    public:

        template<class ContainerT>
        class IterableEnsembleT {
        public:
            IterableEnsembleT(ContainerT &elements) : m_empty_container(nullptr), m_elements(elements) {}

            IterableEnsembleT() : m_empty_container(new ContainerT()), m_elements(*m_empty_container) {}

            virtual ~IterableEnsembleT() { if (m_empty_container) delete m_empty_container; }

            typedef typename ContainerT::iterator iterator;
            typedef typename ContainerT::const_iterator const_iterator;

            iterator begin() { return m_elements.begin(); }

            const_iterator begin() const { return m_elements.begin(); }

            iterator end() { return m_elements.end(); }

            const_iterator end() const { return m_elements.end(); }

            int size() { return m_elements.size(); }

            int size() const { return m_elements.size(); }

        private:
            ContainerT *m_empty_container;
            ContainerT &m_elements;
        };

    public :
        using VertexTypeRef = std::reference_wrapper<VertexType>;
        using VertexTypeConstRef = std::reference_wrapper<const VertexType>;
        using EdgeTypeRef = std::reference_wrapper<EdgeType>;
        using EdgeTypeConstRef = std::reference_wrapper<const EdgeType>;
        using VertexList = std::list<VertexType>;
        using EdgeList = std::list<EdgeType>;
        using VertexTypeRefArray = std::vector<VertexTypeRef>; // todo was a shared Array : see why
        using VertexTypeConstRefArray = std::vector<VertexTypeConstRef>;
        using EdgeTypeRefArray = std::vector<EdgeTypeRef>;
        using EdgeTypeConstRefArray = std::vector<EdgeTypeConstRef>;
        using AdjacencyListType = std::map<VertexTypeConstRef, std::pair<VertexTypeRefArray, EdgeTypeRefArray>>;
        using VertexPair = std::pair<VertexTypeRef, VertexTypeRef>;
        using EdgeToVertexMap = std::map<EdgeTypeConstRef, VertexPair> ;

        using VertexSet = IterableEnsembleT<VertexList>;
        using EdgeSet   =  IterableEnsembleT<EdgeList>;
        using ConnectedEdgeSet = IterableEnsembleT<EdgeTypeRefArray>;

    public:

        using VertexRef = VertexType;
        using EdgeRef = EdgeType;

        // TODO Array de reference_wrapper ne fonctionne pas ...car il fait des T()...voir avec std::vector...

        //! Les arêtes multiples (constituées des mêmes noeuds source et target) ne sont pas autorisées (throw FatalErrorException)
        void addEdge(const VertexType &source_vertex, const VertexType &target_vertex,
                     const EdgeType &source_to_target_edge) {
            _addEdge(source_vertex, target_vertex, source_to_target_edge);
        }

        void addEdge(VertexType &&source_vertex, VertexType &&target_vertex, EdgeType &&source_to_target_edge) {
            _addEdge(source_vertex, target_vertex, source_to_target_edge);
        }

        template<class Vertex, class Edge>
        void _addEdge(Vertex source_vertex, Vertex target_vertex, Edge source_to_target_edge) {
            bool has_edge = (_getEdgeIndex(source_vertex, target_vertex).first != -1 ||
                             m_edge_to_vertex_map.find(source_to_target_edge) != m_edge_to_vertex_map.end());
            if (has_edge)
                throw std::runtime_error{"Cannot insert existing edge."}; // TODO print edge and vertices values if possible (enable_if)
            m_edges.push_back(source_to_target_edge);
            EdgeType &inserted_edge = m_edges.back(); // Get a reference to the inserted objects (since objects are only stored in list, other structures handle references)
            VertexType &inserted_source_vertex = _addVertex(source_vertex);
            VertexType &inserted_target_vertex = _addVertex(target_vertex);
            // Fill adjacency map [source_vertex] = pair<TargetVertexArray,EdgeArray>
            auto adjacency_entry = m_adjacency_list[inserted_source_vertex];
            adjacency_entry.first.push_back(inserted_target_vertex);
            adjacency_entry.second.push_back(inserted_edge);
            // Fill transposed adjacency map [target_vertex] = pair<SourceVertexArray,EdgeArray>
            auto transposed_adjacency_entry = m_adjacency_list_transposed[inserted_target_vertex];
            transposed_adjacency_entry.first.push_back(inserted_source_vertex);
            transposed_adjacency_entry.second.push_back(inserted_edge);
            // Fill edge map [edge] = pair <Vertex,Vertex>
            m_edge_to_vertex_map.insert(std::make_pair(std::ref(inserted_edge),
                                                       std::make_pair(std::ref(inserted_source_vertex),
                                                                      std::ref(inserted_target_vertex))));
            // c'est moche mais on ne peut pas utiliser [] de la map avec reference_wrapper (not default constructible) ni utiliser emplace (pas supporté dans gcc 4.7.2
//    m_edge_to_vertex_map.emplace(std::cref(inserted_edge),std::make_pair(inserted_source_vertex,inserted_target_vertex)); // No Gcc 4,7,2
        }

        //! Renvoie un pointeur vers l'instance d'EdgeType stockée dans le graphe ou nullptr si non trouvé.
        EdgeType *getEdge(const VertexType &source_vertex, const VertexType &target_vertex) {
            return _getEdge(source_vertex, target_vertex);
        }

        //! Renvoie un pointeur vers l'instance d'EdgeType stockée dans le graphe ou nullptr si non trouvé.
        const EdgeType *getEdge(const VertexType &source_vertex, const VertexType &target_vertex) const {
            return _getEdge(source_vertex, target_vertex);
        }

        EdgeType *_getEdge(const VertexType &source_vertex, const VertexType &target_vertex) {
            int edge_index;
            EdgeTypeRefArray edge_array;
            std::tie(edge_index, edge_array) = _getEdgeIndex(source_vertex, target_vertex);
            if (edge_index == -1) return nullptr;
            else return &edge_array[edge_index].get();
        }

        // Implémenter in_edges(vertex) et out_edges(vertex) avec un itérateur...puis edges() et vertices()

        VertexType *getSourceVertex(const EdgeType &edge) {
            typename EdgeToVertexMap::iterator edge_entry = m_edge_to_vertex_map.find(edge);
            if (edge_entry != m_edge_to_vertex_map.end()) return &(edge_entry->second.first.get());
            else return nullptr;
        }

        const VertexType *getSourceVertex(const EdgeType &edge) const {
            auto edge_entry = m_edge_to_vertex_map.find(edge);
            if (edge_entry != m_edge_to_vertex_map.end()) return &edge_entry->second.first.get();
            else return nullptr;
        }

        VertexType *getTargetVertex(const EdgeType &edge) {
            auto edge_entry = m_edge_to_vertex_map.find(edge);
            if (edge_entry != m_edge_to_vertex_map.end()) return &edge_entry->second.second.get();
            else return nullptr;
        }

        const VertexType *getTargetVertex(const EdgeType &edge) const {
            auto edge_entry = m_edge_to_vertex_map.find(edge);
            if (edge_entry != m_edge_to_vertex_map.end()) return &edge_entry->second.second.get();
            else return nullptr;
        }

        VertexSet vertices() { return VertexSet(m_vertices); }

        EdgeSet edges() { return EdgeSet(m_edges); }

        ConnectedEdgeSet inEdges(const VertexType &vertex) {
            auto found_vertex = m_adjacency_list_transposed.find(vertex);
            if (found_vertex == m_adjacency_list_transposed.end()) {
                return ConnectedEdgeSet();
            } else return ConnectedEdgeSet(found_vertex->second.second); // map <vertex, pair <VertexArray, EdgeArray> >
        }

        ConnectedEdgeSet outEdges(const VertexType &vertex) {
            auto found_vertex = m_adjacency_list.find(vertex);
            if (found_vertex == m_adjacency_list.end()) {
                return ConnectedEdgeSet();
            } else return ConnectedEdgeSet(found_vertex->second.second); // map <vertex, pair <VertexArray, EdgeArray> >
        }

    protected:
        VertexList m_vertices;
        EdgeList m_edges;
        AdjacencyListType m_adjacency_list; //! source_vertex -> target_vertices
        AdjacencyListType m_adjacency_list_transposed; //! target_vertex -> source_vertices
        EdgeToVertexMap m_edge_to_vertex_map;

    private:

        template<class Vertex>
        VertexType &_addVertex(Vertex vertex) // to handle _add(VertexType&) et _add(VertexType&&)
        {
            // Look up if vertex does exist
            auto found_vertex = std::find_if(m_vertices.begin(), m_vertices.end(), [&vertex](const VertexType &u) {
                return (!(u < vertex) && !(vertex < u));
            }); // Unary predicate used to avoid contraining VertexObject to be Equality Comparable objects
            if (found_vertex == m_vertices.end()) // Vertex does not exist
            {
                m_vertices.push_back(vertex);
                return m_vertices.back();
            } else return *found_vertex;
        }

        template<class Vertex>
        // to handle Vertex&& et Vertex& = another way to do so ?
        std::pair<int, EdgeTypeRefArray> _getEdgeIndex(Vertex source_vertex, Vertex target_vertex) {
            typename AdjacencyListType::iterator found_source_vertex = m_adjacency_list.find(source_vertex);
            if (found_source_vertex == m_adjacency_list.end()) return std::make_pair(-1, EdgeTypeRefArray());
            int target_vertex_index = _getTargetVertexIndex(found_source_vertex, target_vertex);
            return std::make_pair(target_vertex_index,
                                  found_source_vertex->second.second); // pair < u, pair <[u], [u_v] > >...Use get<T> with pair when available to improve readability
        }

        template<class Vertex>
        // c'est contagieux...
        int _getTargetVertexIndex(typename AdjacencyListType::iterator source_vertex_map_entry, Vertex target_vertex) {
            if (source_vertex_map_entry == m_adjacency_list.end()) return -1;
            return _getConnectedVertexIndex(source_vertex_map_entry, target_vertex);
        }

        template<class Vertex>
        // c'est contagieux...
        int _getConnectedVertexIndex(typename AdjacencyListType::iterator vertex_map_entry, Vertex connected_vertex) {
            VertexTypeRefArray &vertex_array = vertex_map_entry->second.first;
            std::vector<Neo::utils::Int32> indexes(vertex_array.size());
            std::iota(indexes.begin(), indexes.end(), 0);
            auto connected_vertex_index = std::find_if(indexes.begin(), indexes.end(),
                                                       [&](const int index) {
                                                           return (!(vertex_array[index].get() < connected_vertex) &&
                                                                   !(connected_vertex < vertex_array[index].get()));
                                                       });
            if (connected_vertex_index == indexes.end()) return -1;
            else return *connected_vertex_index;
        }

    };

} // namespace Nep
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



#endif //NEO_GRAPHBASE_H
