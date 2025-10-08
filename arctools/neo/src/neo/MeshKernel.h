// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshKernel                                     (C) 2000-2025             */
/*                                                                           */
/* Brief code description                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef NEO_MESHKERNEL_H
#define NEO_MESHKERNEL_H

#include <string>

#include "neo/Neo.h"
#include "sgraph/DirectedAcyclicGraph.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Neo
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace tye
{ // type engine : tool to visit variant

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename... T> struct VisitorOverload : public T...
  {
    using T::operator()...;
  };

  /*---------------------------------------------------------------------------*/

  template <typename Func, typename Variant>
  void apply(Func& func, Variant& arg) {
#ifdef _MSC_VER
    auto default_func = [](auto arg) {
#else
    auto default_func = []([[maybe_unused]] auto arg) {
#endif
      std::cout << "Wrong Property Type" << std::endl;
    }; // todo: prevent this behavior (statically ?)
    std::visit(VisitorOverload{ default_func, func }, arg);
  }

  /*---------------------------------------------------------------------------*/

  template <typename Func, typename Variant>
  void apply(Func& func, Variant& arg1, Variant& arg2) {
    std::visit([&arg2, &func](auto& concrete_arg1) {
      std::visit([&concrete_arg1, &func](auto& concrete_arg2) {
#ifdef _MSC_VER // incomplete maybe_unused support on visual (v 19.29)
        auto functor = VisitorOverload{ [](const auto& arg1, const auto& arg2) { std::cout << "### WARNING: Algorithm not found. You may have missed its signature ###" << std::endl; }, func }; // todo: prevent this behavior (statically ?)
#else
                                                       auto functor = VisitorOverload{ []([[maybe_unused]] const auto& arg1, [[maybe_unused]] const auto& arg2) { std::cout << "### WARNING: Algorithm not found. You may have missed its signature ###" << std::endl; }, func }; // todo: prevent this behavior (statically ?)
#endif
        functor(concrete_arg1, concrete_arg2); // arg1 & arg2 are variants, concrete_arg* are concrete arguments
      },
                 arg2);
    },
               arg1);
  }

  /*---------------------------------------------------------------------------*/

  template <typename Func, typename Variant>
  void apply(Func& func, Variant& arg1, Variant& arg2, Variant& arg3) {
    std::visit([&arg2, &arg3, &func](auto& concrete_arg1) {
      std::visit([&concrete_arg1, &arg3, &func](auto& concrete_arg2) {
        std::visit([&concrete_arg1, &concrete_arg2, &func](auto& concrete_arg3) {
#ifdef _MSC_VER
          auto functor = VisitorOverload{ [](const auto& arg1, const auto& arg2, const auto& arg3) { std::cout << "### WARNING: Algorithm not found. You may have missed its signature ###" << std::endl; }, func }; // todo: prevent this behavior (statically ?)
#else
                                                                                                                               auto functor = VisitorOverload{ []([[maybe_unused]] const auto& arg1, [[maybe_unused]] const auto& arg2, [[maybe_unused]] const auto& arg3) { std::cout << "### WARNING: Algorithm not found. You may have missed its signature ###" << std::endl; }, func }; // todo: prevent this behavior (statically ?)
#endif
          functor(concrete_arg1, concrete_arg2, concrete_arg3); // arg1 & arg2 are variants, concrete_arg* are concrete arguments
        },
                   arg3);
      },
                 arg2);
    },
               arg1);
  }

  /*---------------------------------------------------------------------------*/

  // template deduction guides
  template <typename... T> VisitorOverload(T...) -> VisitorOverload<T...>;

} // namespace tye

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MeshKernel
{
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  struct PropertyHolder
  {
    Family& m_family;
    std::string m_name;
    PropertyStatus m_status = PropertyStatus::ComputedProperty;

    auto& operator()() {
      return m_family.getProperty(m_name);
    }

    std::string uniqueName() const noexcept {
      return m_name + "_" + m_family.name();
    }
  };

  /*---------------------------------------------------------------------------*/

  struct InProperty : public PropertyHolder
  {};

  /*---------------------------------------------------------------------------*/

  struct OutProperty : public PropertyHolder
  {};

  /*---------------------------------------------------------------------------*/

  struct IAlgorithm
  {
    virtual void operator()() = 0;
    virtual InProperty const& inProperty(int index = 0) const = 0;
    virtual OutProperty const& outProperty(int index = 0) const = 0;
    virtual ~IAlgorithm() = default;
    virtual int nbInProperties() const = 0;
    virtual int nbOutProperties() const = 0;
  };

  /*---------------------------------------------------------------------------*/

  template <typename Algorithm>
  struct AlgoHandler : public IAlgorithm
  {
    AlgoHandler(InProperty&& in_prop, OutProperty&& out_prop, Algorithm&& algo)
    : m_in_property(std::move(in_prop))
    , m_out_property(std::move(out_prop))
    , m_algo(std::forward<Algorithm>(algo)) {}
    InProperty m_in_property;
    OutProperty m_out_property;
    Algorithm m_algo;
    void operator()() override {
      tye::apply(m_algo, m_in_property(), m_out_property());
    }
    InProperty const& inProperty(int index) const override {
      if (index == 0)
        return m_in_property;
      else
        throw std::invalid_argument("The current algo has only one inProperty. Cannot call IAlgorithm::inProperty(index) with index > 0");
    }
    OutProperty const& outProperty(int index) const override {
      if (index == 0)
        return m_out_property;
      else
        throw std::invalid_argument("The current algo has only one outProperty. Cannot call IAlgorithm::outProperty(index) with index > 0");
    }

    int nbInProperties() const override {
      return 1;
    }

    int nbOutProperties() const override {
      return 1;
    }
  };

  /*---------------------------------------------------------------------------*/

  template <typename Algorithm>
  struct DualInAlgoHandler : public IAlgorithm
  {
    DualInAlgoHandler(InProperty&& in_prop1, InProperty&& in_prop2, OutProperty&& out_prop, Algorithm&& algo)
    : m_in_property1(std::move(in_prop1))
    , m_in_property2(std::move(in_prop2))
    , m_out_property(std::move(out_prop))
    , m_algo(std::forward<Algorithm>(algo)) {}
    InProperty m_in_property1;
    InProperty m_in_property2;
    OutProperty m_out_property;
    Algorithm m_algo;
    void operator()() override {
      tye::apply(m_algo, m_in_property1(), m_in_property2(), m_out_property());
    }
    InProperty const& inProperty(int index) const override {
      if (index == 0)
        return m_in_property1;
      else if (index == 1)
        return m_in_property2;
      else
        throw std::invalid_argument("The current algo has only two inProperty. Cannot call IAlgorithm::inProperty(index) with index > 1");
    }
    OutProperty const& outProperty(int index) const override {
      if (index == 0)
        return m_out_property;
      else
        throw std::invalid_argument("The current algo has only one outProperty. Cannot call IAlgorithm::outProperty(index) with index > 0");
    }

    int nbInProperties() const override {
      return 2;
    }

    int nbOutProperties() const override {
      return 1;
    }
  };

  /*---------------------------------------------------------------------------*/

  template <typename Algorithm>
  struct DualOutAlgoHandler : public IAlgorithm
  {
    DualOutAlgoHandler(InProperty&& in_prop, OutProperty&& out_prop1, OutProperty&& out_prop2, Algorithm&& algo)
    : m_in_property(std::move(in_prop))
    , m_out_property1(std::move(out_prop1))
    , m_out_property2(std::move(out_prop2))
    , m_algo(std::forward<Algorithm>(algo)) {}
    InProperty m_in_property;
    OutProperty m_out_property1;
    OutProperty m_out_property2;
    Algorithm m_algo;
    void operator()() override {
      tye::apply(m_algo, m_in_property(), m_out_property1(), m_out_property2());
    }
    InProperty const& inProperty(int index) const override {
      if (index == 0)
        return m_in_property;
      else
        throw std::invalid_argument("The current algo has only one inProperty. Cannot call IAlgorithm::inProperty(index) with index > 0");
    }
    OutProperty const& outProperty(int index) const override {
      if (index == 0)
        return m_out_property1;
      else if (index == 1)
        return m_out_property2;
      else
        throw std::invalid_argument("The current algo has only two outProperty. Cannot call IAlgorithm::outProperty(index) with index > 1");
    }

    int nbInProperties() const override {
      return 1;
    }

    int nbOutProperties() const override {
      return 2;
    }
  };

  /*---------------------------------------------------------------------------*/

  template <typename Algorithm>
  struct NoDepsAlgoHandler : public IAlgorithm
  {
    NoDepsAlgoHandler(OutProperty&& out_prop, Algorithm&& algo)
    : m_out_property(std::move(out_prop))
    , m_algo(std::forward<Algorithm>(algo)) {}
    OutProperty m_out_property;
    Algorithm m_algo;
    void operator()() override {
      tye::apply(m_algo, m_out_property());
    }
    InProperty const& inProperty(int) const override {
      throw std::invalid_argument("The current algo has no inProperty. Cannot call IAlgorithm::inProperty(index)");
    }
    OutProperty const& outProperty(int index) const override {
      if (index == 0)
        return m_out_property;
      else
        throw std::invalid_argument("The current algo has only one outProperty. Cannot call IAlgorithm::outProperty(index) with index > 0");
    }

    int nbInProperties() const override {
      return 0;
    }

    int nbOutProperties() const override {
      return 1;
    }
  };

  /*---------------------------------------------------------------------------*/

  template <typename Algorithm>
  struct NoDepsDualOutAlgoHandler : public IAlgorithm
  {
    NoDepsDualOutAlgoHandler(OutProperty&& out_prop1, OutProperty&& out_prop2, Algorithm&& algo)
    : m_out_property1(std::move(out_prop1))
    , m_out_property2(std::move(out_prop2))
    , m_algo(std::forward<Algorithm>(algo)) {}
    OutProperty m_out_property1;
    OutProperty m_out_property2;
    Algorithm m_algo;
    void operator()() override {
      tye::apply(m_algo, m_out_property1(), m_out_property2());
    }
    InProperty const& inProperty(int) const override {
      throw std::invalid_argument("The current algo has no inProperty. Cannot call IAlgorithm::inProperty(index)");
    }
    OutProperty const& outProperty(int index) const override {
      if (index == 0)
        return m_out_property1;
      else if (index == 1)
        return m_out_property2;
      else
        throw std::invalid_argument("The current algo has only two outProperty. Cannot call IAlgorithm::inProperty(index) with index > 1");
    }

    int nbInProperties() const override {
      return 0;
    }

    int nbOutProperties() const override {
      return 2;
    }
  };

  /*---------------------------------------------------------------------------*/

#ifdef _MSC_VER
  struct PropertyHolderLessComparator {
      bool operator() (PropertyHolder const& a, PropertyHolder const& b) const { return a.uniqueName() < b.uniqueName(); }
  };
#endif

  class AlgorithmPropertyGraph
  {
   public:
    std::string m_name;
    int m_rank = 0;
    std::list<std::shared_ptr<IAlgorithm>> m_algos;
    using AlgoPtr = std::shared_ptr<IAlgorithm>;
    using ProducingAlgoArray = std::vector<AlgoPtr>;
    using ConsumingAlgoArray = std::vector<AlgoPtr>;
    using PropertyRef = std::reference_wrapper<const PropertyHolder>;
#ifndef _MSC_VER
    static inline auto m_prop_holder_less_comparator = [](PropertyHolder const& a, PropertyHolder const& b) { return a.uniqueName() < b.uniqueName(); };
    using PropertyHolderLessComparator = decltype(m_prop_holder_less_comparator);
#else
    PropertyHolderLessComparator  m_prop_holder_less_comparator;
#endif
    std::map<PropertyHolder, std::pair<ProducingAlgoArray, ConsumingAlgoArray>, PropertyHolderLessComparator> m_property_algorithms{m_prop_holder_less_comparator };
    SGraph::DirectedAcyclicGraph<AlgoPtr, PropertyHolder> m_dag;
    std::list<AlgoPtr> m_kept_in_out_algos;
    std::list<AlgoPtr> m_kept_out_algos;
    std::list<AlgoPtr> m_kept_dual_out_algos;
    std::list<AlgoPtr> m_kept_dual_in_algos;
    std::list<AlgoPtr> m_kept_no_deps_dual_out_algos;
    enum class AlgorithmExecutionOrder
    {
      FIFO,
      LIFO,
      DAG
    };
    enum class AlgorithmPersistence
    {
      DropAfterExecution,
      KeepAfterExecution
    };

   public:

    template <typename Algorithm>
    void addAlgorithm(InProperty&& in_property, OutProperty&& out_property, Algorithm algo, AlgorithmPersistence persistance = AlgorithmPersistence::DropAfterExecution) { // problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
      auto algo_handler = std::make_shared<AlgoHandler<decltype(algo)>>(std::move(in_property), std::move(out_property), std::forward<Algorithm>(algo));
      m_algos.push_back(algo_handler);
      _addAlgoFromHandler(algo_handler);
      if (persistance == AlgorithmPersistence::KeepAfterExecution)
        m_kept_in_out_algos.push_back(algo_handler);
    }

    template <typename Algorithm>
    void addAlgorithm(OutProperty&& out_property, Algorithm algo, AlgorithmPersistence persistance = AlgorithmPersistence::DropAfterExecution) { // problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
      auto algo_handler = std::make_shared<NoDepsAlgoHandler<decltype(algo)>>(std::move(out_property), std::forward<Algorithm>(algo));
      m_algos.push_back(algo_handler);
      _addAlgoFromNoDepsHandler(algo_handler);
      if (persistance == AlgorithmPersistence::KeepAfterExecution)
        m_kept_out_algos.push_back(algo_handler);
    }

    template <typename Algorithm>
    void addAlgorithm(InProperty&& in_property1, InProperty&& in_property2, OutProperty&& out_property, Algorithm algo, AlgorithmPersistence persistance = AlgorithmPersistence::DropAfterExecution) { // problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
      auto algo_handler = std::make_shared<DualInAlgoHandler<decltype(algo)>>(
      std::move(in_property1),
      std::move(in_property2),
      std::move(out_property),
      std::forward<Algorithm>(algo));
      m_algos.push_back(algo_handler);
      _addAlgoFromDualInHandler(algo_handler);
      if (persistance == AlgorithmPersistence::KeepAfterExecution)
        m_kept_dual_in_algos.push_back(algo_handler);
    }

    template <typename Algorithm>
    void addAlgorithm(InProperty&& in_property, OutProperty&& out_property1, OutProperty&& out_property2, Algorithm algo, AlgorithmPersistence persistance = AlgorithmPersistence::DropAfterExecution) { // problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
      auto algo_handler = std::make_shared<DualOutAlgoHandler<decltype(algo)>>(std::move(in_property), std::move(out_property1), std::move(out_property2), std::forward<Algorithm>(algo));
      m_algos.push_back(algo_handler);
      _addAlgoFromDualOutHandler(algo_handler);
      if (persistance == AlgorithmPersistence::KeepAfterExecution)
        m_kept_dual_out_algos.push_back(algo_handler);
    }

    template <typename Algorithm>
    void addAlgorithm(OutProperty&& out_property1, OutProperty&& out_property2, Algorithm algo, AlgorithmPersistence persistance = AlgorithmPersistence::DropAfterExecution) { // problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
      auto algo_handler = std::make_shared<NoDepsDualOutAlgoHandler<decltype(algo)>>(std::move(out_property1), std::move(out_property2), std::forward<Algorithm>(algo));
      m_algos.push_back(algo_handler);
      _addAlgoFromNoDepsDualOutHandler(algo_handler);
      if (persistance == AlgorithmPersistence::KeepAfterExecution)
        m_kept_no_deps_dual_out_algos.push_back(algo_handler);
    }

    /*!
    * @brief Remove all added algorithms, only not-persistent by default
    * @param remove_kept_algorithm : to choose if algorithms added with AlgorithmPersistence::KeepAfterExecution are removed or not
    */
    void removeAlgorithms(bool remove_kept_algorithm = false) {
      m_algos.clear();
      m_property_algorithms.clear();
      m_dag.clear();
      if (remove_kept_algorithm) {
        m_kept_in_out_algos.clear();
        m_kept_dual_in_algos.clear();
        m_kept_out_algos.clear();
        m_kept_dual_out_algos.clear();
        m_kept_no_deps_dual_out_algos.clear();
      }
      else {
        _addKeptAlgorithms(); // Kept algorithms are rescheduled
      }
    }

    /*!
     * @brief Apply added algorithms
     * @param execution_order : to choose between LIFO, FIFO or DAG
     * @return object EndOfMeshUpdate to unlock FutureItemRange
     * Added algorithms are removed at the end of the method
     */
    EndOfMeshUpdate applyAlgorithms(AlgorithmExecutionOrder execution_order = AlgorithmExecutionOrder::DAG) {
      return _applyAlgorithms(execution_order, false);
    }

    /*!
     * @brief Apply added algorithms
     * @param execution_order
     * @return object EndOfMeshUpdate to unlock FutureItemRange
     * Added algorithms are kept at the end of the method. If the method is called twice, the algorithms are played again.
     */
    EndOfMeshUpdate applyAndKeepAlgorithms(AlgorithmExecutionOrder execution_order = AlgorithmExecutionOrder::DAG) {
      return _applyAlgorithms(execution_order, true);
    }

    EndOfMeshUpdate _applyAlgorithms(AlgorithmExecutionOrder execution_order, bool do_keep_algorithms) {
      Neo::print(m_rank) << "-- apply added algorithms with execution order ";
      switch (execution_order) {
      case AlgorithmExecutionOrder::FIFO:
        Neo::print(m_rank) << "FIFO --" << std::endl;
        std::for_each(m_algos.begin(), m_algos.end(), [](auto& algo) { (*algo.get())(); });
        break;
      case AlgorithmExecutionOrder::LIFO:
        Neo::print(m_rank) << "LIFO --" << std::endl;
        std::for_each(m_algos.rbegin(), m_algos.rend(), [](auto& algo) { (*algo.get())(); });
        break;
      case AlgorithmExecutionOrder::DAG:
        Neo::print(m_rank) << "DAG --" << std::endl;
        _build_graph();
        try {
          auto sorted_graph = m_dag.topologicalSort();
          std::for_each(sorted_graph.begin(), sorted_graph.end(), [](auto& algo) { (*algo.get())(); });
        }
        catch (std::runtime_error& error) {
          if (!do_keep_algorithms)
            removeAlgorithms();
          throw error;
        }
        break;
      }
      if (!do_keep_algorithms)
        removeAlgorithms();
      return EndOfMeshUpdate{};
    }

    PropertyStatus propertyStatus(std::string const& property_unique_name,std::shared_ptr<IAlgorithm>const& algorithm) {
      for (auto input_property_index = 0; input_property_index < algorithm->nbInProperties(); ++input_property_index) {
        if (property_unique_name == algorithm->inProperty(input_property_index).uniqueName()) {
          return algorithm->inProperty(input_property_index).m_status;
        }
      }
      for (auto output_property_index = 0; output_property_index < algorithm->nbOutProperties(); ++output_property_index) {
        if (property_unique_name == algorithm->outProperty(output_property_index).uniqueName()) {
          return algorithm->outProperty(output_property_index).m_status;
        }
      }
    }

  private:

    void _build_graph() {
      // Mark algorithms that won't have at least one input property
      auto compare_algo = [](auto const& algo1, auto const& algo2) { return algo1.get() < algo2.get(); };
      std::set<AlgoPtr,decltype(compare_algo)> to_remove_algos{compare_algo};
      for (auto&& [property, property_algos] : m_property_algorithms) {
        auto& [producing_property_array, consuming_property_array] = property_algos;
        if (!property.m_family.hasProperty(property.m_name)) {
          // Property does not exist: Remove all its algorithms
          for (auto&& algo_to_remove : consuming_property_array) {
            to_remove_algos.insert(algo_to_remove);
          }
          for (auto&& algo_to_remove : producing_property_array) { // property does not exist
            to_remove_algos.insert(algo_to_remove);
          }
        }
        else if (producing_property_array.size() == 0) {
          // The property has no producing algorithm: the algorithm consuming it as a ComputedProperty must be removed
          for (auto&& consuming_algo : consuming_property_array) {
            // Warning, cannot use the status of the property: this status may change with algorithm and property only stored once in m_property_algorithm
            if (propertyStatus(property.uniqueName(),consuming_algo) == PropertyStatus::ComputedProperty) to_remove_algos.insert(consuming_algo);
          }
        }
      }
      // Handle removed algos: they don't produce their output => new algos may be to remove
      std::map<std::string, int> property_nb_producing_algos;
      std::vector<AlgoPtr> removed_algos(to_remove_algos.begin(),to_remove_algos.end()), currently_removed_algos;
      while (removed_algos.size() > 0) {
        for (auto&& algo_to_remove : removed_algos) {
          for (auto out_prop_index = 0; out_prop_index < algo_to_remove->nbOutProperties(); ++out_prop_index) {
            auto const& out_prop = algo_to_remove->outProperty(out_prop_index);
            auto& [producing_property_array, consuming_property_array] = m_property_algorithms[out_prop];
            auto prop_nb_prod_algo_it = property_nb_producing_algos.find(out_prop.uniqueName());
            if (prop_nb_prod_algo_it == property_nb_producing_algos.end()) { // property is not yet registered
              property_nb_producing_algos[out_prop.uniqueName()] = producing_property_array.size() - 1; // one of its production algo is removed
            }
            else {
              --prop_nb_prod_algo_it->second;
            }
            prop_nb_prod_algo_it = property_nb_producing_algos.find(out_prop.uniqueName()); // iterator may be invalidated by insertion

            if (prop_nb_prod_algo_it->second == 0) { // property has no more producing algo
              for (auto&& consuming_algo : consuming_property_array) { // property does not exist
                if (propertyStatus(out_prop.uniqueName(),consuming_algo) == PropertyStatus::ComputedProperty) {
                  to_remove_algos.insert(consuming_algo);
                  currently_removed_algos.push_back(consuming_algo);
                }
              }
            }
          }
        }
        removed_algos = currently_removed_algos;
        currently_removed_algos.clear();
      }

      // Add edges between producing and consuming algos
      auto is_removed_algo = [&to_remove_algos](AlgoPtr const& algo) {
        return (to_remove_algos.find(algo) != to_remove_algos.end());
      };

      for (auto& [property, property_algos] : m_property_algorithms) {
        if (!property.m_family.hasProperty(property.m_name))
          continue;
        auto& [producing_property_array, consuming_property_array] = property_algos;
        for (auto& producing_algo : producing_property_array) {
          // Add each producing algo in graph: it may have no consuming algo
          if (!is_removed_algo(producing_algo))
            m_dag.addVertex(producing_algo);
          for (auto& consuming_algo : consuming_property_array) {
            if (!is_removed_algo(producing_algo) && !is_removed_algo(consuming_algo))
              m_dag.addEdge(producing_algo, consuming_algo, property);
          }
        }
      }
    }

    void _addAlgoFromHandler(std::shared_ptr<IAlgorithm> algo_handler) {
      _addProducingAlgo(algo_handler->outProperty(), algo_handler);
      _addConsumingAlgo(algo_handler->inProperty(), algo_handler);
    }

    void _addAlgoFromNoDepsHandler(std::shared_ptr<IAlgorithm> algo_handler) {
      _addProducingAlgo(algo_handler->outProperty(), algo_handler);
    }

    void _addAlgoFromDualInHandler(std::shared_ptr<IAlgorithm> algo_handler) {
      _addProducingAlgo(algo_handler->outProperty(), algo_handler);
      _addConsumingAlgo(algo_handler->inProperty(0), algo_handler);
      _addConsumingAlgo(algo_handler->inProperty(1), algo_handler);
    }

    void _addAlgoFromDualOutHandler(std::shared_ptr<IAlgorithm> algo_handler) {
      _addProducingAlgo(algo_handler->outProperty(0), algo_handler);
      _addProducingAlgo(algo_handler->outProperty(1), algo_handler);
      _addConsumingAlgo(algo_handler->inProperty(), algo_handler);
    }

    void _addAlgoFromNoDepsDualOutHandler(std::shared_ptr<IAlgorithm> algo_handler) {
      _addProducingAlgo(algo_handler->outProperty(0), algo_handler);
      _addProducingAlgo(algo_handler->outProperty(1), algo_handler);
    }

    void _addProducingAlgo(OutProperty const& out_property, std::shared_ptr<IAlgorithm> algo) {
      // add algo as one of producing algo of out_property
      auto& [producing_algo_array, consuming_algo_array] = m_property_algorithms[out_property];
      producing_algo_array.push_back(algo);
    }

    void _addConsumingAlgo(InProperty const& in_property, std::shared_ptr<IAlgorithm> algo) {
      // add algo as one of the consuming algos of out_property
      auto& [producing_algo_array, consuming_algo_array] = m_property_algorithms[in_property];
      consuming_algo_array.push_back(algo);
    }

    void _addKeptAlgorithms() {
      for (auto& kept_algo : m_kept_in_out_algos) {
        _addAlgoFromHandler(kept_algo);
      }
      for (auto& kept_dual_in_algo : m_kept_dual_in_algos) {
        _addAlgoFromDualInHandler(kept_dual_in_algo);
      }
      for (auto& kept_out_algo : m_kept_out_algos) {
        _addAlgoFromNoDepsHandler(kept_out_algo);
      }
      for (auto& kept_dual_out_algo : m_kept_dual_out_algos) {
        _addAlgoFromDualOutHandler(kept_dual_out_algo);
      }
      for (auto& kept_dual_out_algo : m_kept_no_deps_dual_out_algos) {
        _addAlgoFromNoDepsDualOutHandler(kept_dual_out_algo);
      }
    }
  };

  /*---------------------------------------------------------------------------*/

} // namespace MeshKernel

/*---------------------------------------------------------------------------*/

} // namespace Neo

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_MESHKERNEL_H
