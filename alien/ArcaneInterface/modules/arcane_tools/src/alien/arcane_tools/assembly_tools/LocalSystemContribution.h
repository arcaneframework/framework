/*
 * LocalSystemContribution.h
 *
 *  Created on: 20 avr. 2021
 *      Author: gratienj
 */

#pragma once

#include <unordered_map>

#include <Eigen/Dense>

namespace Alien {

namespace ArcaneTools
{

  template<typename ValueT=Arccore::Real, typename IndexT=Arccore::Integer,int dim_value=3>
  class BoundaryConditionMng
  {
  public :
    static const int        dim = dim_value ;
    typedef IndexT          index_type ;
    typedef ValueT          value_type ;
    typedef ValueT          BCValueType ;
    typedef enum {
      Dirichlet,
      Neumann,
      Undefined
    } eBCType;

    Arccore::UniqueArray<index_type>                           m_dof_is_boundary;
    std::map<index_type, std::array<eBCType,dim>>             m_dof_bc_type ;
    std::map<index_type, std::array<BCValueType,dim>>         m_dof_bc_values ;


    bool isBoundary(index_type dof_lid) const {
      return m_dof_is_boundary[dof_lid] == 1 ;
    }

    eBCType bcType(index_type dof_lid, index_type u_id)
    {
      assert(m_dof_is_boundary[dof_lid==1]) ;
      assert( (u_id>=0) && (u_id<dim)) ;
      return m_dof_bc_type[dof_lid][u_id] ;
    }
  };

  template<typename ValueT=Arccore::Real, typename IndexT=Arccore::Integer,int dim_value=3>
  class SystemContributionMng
  {
  public:

    static const int                                          dim = dim_value ;
    typedef IndexT                                            index_type ;
    typedef ValueT                                            value_type ;

    typedef Alien::ArcaneTools::IIndexManager                 IndexManagerType;
    typedef Alien::ArcaneTools::IIndexManager::ScalarIndexSet ScalarIndexSetType;
    typedef Alien::ArcaneTools::IIndexManager::VectorIndexSet VectorIndexSetType;
    typedef BoundaryConditionMng<value_type,index_type,dim>   BoundaryConditionMngType ;

    class LocalSystem
    {
    public :
      static const int                                         dim = dim_value ;
      typedef Eigen::Map<Eigen::MatrixXd>                      EigenMatrixType ;
      typedef Eigen::Map<Eigen::VectorXd>                      EigenVectorType ;

      typedef SystemContributionMng<value_type,index_type,dim> ParentType ;

    private :
      ParentType&             m_parent ;
      Arcane::Cell            m_cell ;
      EigenMatrixType         m_matrix;
      EigenVectorType         m_rhs ;
      Arcane::ItemVectorView  m_dof_vector ;
      index_type              m_null_index = -1 ;

    public :

      LocalSystem(SystemContributionMng& parent, Arcane::Cell const& cell, Arcane::ItemVectorView const& dof_vector)
      : m_parent(parent)
      , m_cell(cell)
      , m_matrix(parent.matrixBuffer().data(),dim*dof_vector.size(),dim*dof_vector.size())
      , m_rhs(parent.rhsBuffer().data(),dim*dof_vector.size())
      , m_dof_vector(dof_vector)
      , m_null_index(parent.m_index_manager->nullIndex())
      {
        m_matrix.setZero() ;
        m_rhs.setZero() ;
      }

      EigenMatrixType& matrix() {
        return m_matrix ;
      }
      EigenVectorType& rhs() {
        return m_rhs ;
      }

      void defineProfile(Alien::MatrixProfiler& profiler)
      {
        for(auto dof1 : m_dof_vector)
        {
          if ( not dof1->isOwn())
            continue;

          const auto dof1_lid = dof1.localId() ;

          bool dof1_is_boundary = m_parent.m_boundary_condition_mng.isBoundary(dof1_lid) ;

          for(auto dof2 : m_dof_vector)
          {
            const auto dof2_lid = dof2.localId() ;

            bool dof2_is_boundary = m_parent.m_boundary_condition_mng.isBoundary(dof2_lid) ;

            for (int ieq = 0; ieq < dim; ++ieq)
            {
              if (dof1_is_boundary && m_parent.m_boundary_condition_mng.bcType(dof1_lid,ieq) == BoundaryConditionMngType::Dirichlet)
                continue;

              const auto index_eq =  m_parent.m_indexes[dof1_lid][ieq] ;
              assert(index_eq != m_null_index) ;
              for (int iuk=0; iuk < dim; ++iuk)
              {
                if( not dof2_is_boundary || m_parent.m_boundary_condition_mng.bcType(dof2_lid,iuk) != BoundaryConditionMngType::Dirichlet)
                {
                  const auto index_uk = m_parent.m_indexes[dof2_lid][iuk] ;
                  assert(index_uk != m_null_index) ;
                  profiler.addMatrixEntry(index_eq, index_uk);
                }
              }
            }
          }
        }
      }

      void assemble (Alien::ProfiledMatrixBuilder& matrix_builder,
                     Alien::VectorWriter& rhs_builder)
      {
        int dof1_index = 0 ;
        for( auto dof1 : m_dof_vector)
        {
          if (!dof1.isOwn())
            continue;

          const auto dof1_lid = dof1.localId() ;
          bool dof1_is_boundary = m_parent.m_boundary_condition_mng.isBoundary(dof1_lid) ;

          const Eigen::Index start1(dim * dof1_index);

          int dof2_index = 0 ;
          for( auto dof2 : m_dof_vector)
          {
            auto dof2_lid = dof2.localId () ;

            bool dof2_is_boundary = m_parent.m_boundary_condition_mng.isBoundary(dof2_lid) ;

            const Eigen::Index start2 (dim * dof2_index);

            for (auto ieq (0); ieq < dim; ++ieq)
            {
                if (dof1_is_boundary && m_parent.m_boundary_condition_mng.bcType(dof1_lid,ieq) == BoundaryConditionMngType::Dirichlet)
                  continue;

                const Eigen::Index eigInd (start1 + ieq);

                const auto index_eq = m_parent.m_indexes[dof1_lid][ieq] ;
                assert(index_eq != m_null_index) ;

                for (auto iuk (0); iuk < dim; ++iuk)
                {
                    const Eigen::Index eigIndb (start2 + iuk);

                    if( not dof2_is_boundary || m_parent.m_boundary_condition_mng.bcType(dof2_lid,iuk) != BoundaryConditionMngType::Dirichlet)
                    {
                      const auto index_uk = m_parent.m_indexes[dof2_lid][iuk] ;
                      assert(index_uk != m_null_index) ;
                      const auto contribution (m_matrix (eigInd, eigIndb));
                      matrix_builder (index_eq, index_uk) += contribution;
                    }
                    else
                    {
                      auto uDD = m_parent.m_boundary_condition_mng.m_dof_bc_values[dof2_lid][iuk];
                      const auto contribution (m_matrix (eigInd, eigIndb) * uDD);
                      rhs_builder[index_eq] -= contribution;
                    }
                }
             }
             ++dof2_index ;
          }
          ++dof1_index ;
        }
      }


      template<typename LambdaT>
      void assemble (Alien::ProfiledMatrixBuilder& matrix_builder,
                     Alien::VectorWriter& rhs_builder,
                     LambdaT& forcing_term)
      {
        int dof1_index = 0 ;
        for( auto dof1 : m_dof_vector)
        {
          if (!dof1.isOwn())
            continue;

          const auto dof1_lid = dof1.localId() ;

          bool dof1_is_boundary = m_parent.m_boundary_condition_mng.isBoundary(dof1_lid) ;

          const Eigen::Index start1(dim * dof1_index);

          for (auto ieq(0); ieq < dim; ++ieq)
          {
            if (dof1_is_boundary && m_parent.m_boundary_condition_mng.bcType(dof1_lid,ieq) == BoundaryConditionMngType::Dirichlet)
              continue;
            const auto index_eq = m_parent.m_indexes[dof1_lid][ieq] ;
            assert(index_eq != m_null_index) ;
            auto contribution = forcing_term(m_cell,dof1,ieq) ;
            rhs_builder[index_eq] += contribution;
          }

          int dof2_index = 0 ;
          for( auto dof2 : m_dof_vector)
          {
            auto dof2_lid = dof2.localId () ;

            bool dof2_is_boundary = m_parent.m_boundary_condition_mng.isBoundary(dof2_lid) ;

            const Eigen::Index start2 (dim * dof2_index);

            for (auto ieq (0); ieq < dim; ++ieq)
            {
                if (dof1_is_boundary && m_parent.m_boundary_condition_mng.bcType(dof1_lid,ieq) == BoundaryConditionMngType::Dirichlet)
                  continue;

                const Eigen::Index eigInd (start1 + ieq);

                const auto index_eq = m_parent.m_indexes[dof1_lid][ieq] ;
                assert(index_eq != m_null_index) ;

                for (auto iuk (0); iuk < dim; ++iuk)
                {
                    const Eigen::Index eigIndb (start2 + iuk);

                    if( not dof2_is_boundary || m_parent.m_boundary_condition_mng.bcType(dof2_lid,ieq) != BoundaryConditionMngType::Dirichlet)
                    {
                      const auto index_uk = m_parent.m_indexes[dof2_lid][iuk] ;
                      assert(index_uk != m_null_index) ;
                      const auto contribution (m_matrix (eigInd, eigIndb));
                      matrix_builder (index_eq, index_uk) += contribution;
                    }
                    else
                    {
                      auto uDD = m_parent.m_boundary_condition_mng.m_dof_bc_values[dof2_lid][iuk];
                      const auto contribution (m_matrix (eigInd, eigIndb) * uDD);
                      rhs_builder[index_eq] -= contribution;
                    }
                }
             }
             ++dof2_index ;
          }
          ++dof1_index ;
        }
      }
    };

    SystemContributionMng (Arccore::Integer maxNumberOfNodesPerCell,
                           Alien::ArcaneTools::IIndexManager* index_manager,
                           VectorIndexSetType& index_sets,
                           Arccore::Array2View<index_type> indexes)
    : m_maxNumberOfNodesPerCell(maxNumberOfNodesPerCell)
    , m_index_manager(index_manager)
    , m_index_sets(index_sets)
    , m_indexes(indexes)
    {
      m_matrix_buffer.resize(dim*dim*m_maxNumberOfNodesPerCell*m_maxNumberOfNodesPerCell)  ;
      m_rhs_buffer.resize(dim*m_maxNumberOfNodesPerCell)  ;
    }

    virtual ~SystemContributionMng () {}

    Arccore::ArrayView<value_type> matrixBuffer() {
      return m_matrix_buffer.view() ;
    }

    Arccore::ArrayView<value_type> rhsBuffer() {
      return m_rhs_buffer.view() ;
    }




  Arccore::Integer                               m_maxNumberOfNodesPerCell = 0 ;
  Arccore::UniqueArray<value_type>               m_matrix_buffer ;
  Arccore::UniqueArray<value_type>               m_rhs_buffer ;
  IndexManagerType*                              m_index_manager = nullptr;
  VectorIndexSetType                             m_index_sets;
  Arccore::Array2View<index_type>                m_indexes;

  BoundaryConditionMngType                       m_boundary_condition_mng ;


  };

} /* namespace AlienTools */

} /* namespace Alien */
