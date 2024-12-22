// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once


#include <vector>
#include <list>
#ifdef ALIEN_USE_COMPOSYX
#include <composyx.hpp>
#include <composyx/part_data/PartMatrix.hpp>
#include <composyx/loc_data/SparseMatrixCSC.hpp>
#include <composyx/solver/ConjugateGradient.hpp>
#include <composyx/solver/BiCGSTAB.hpp>
#include <composyx/solver/GMRES.hpp>
#include <composyx/solver/Jacobi.hpp>
#if defined(COMPOSYX_USE_QRMUMPS)
#include <composyx/solver/QrMumps.hpp>
#endif
#include <composyx/solver/Mumps.hpp>
#include <composyx/solver/Pastix.hpp>

#include <composyx/precond/DiagonalPrecond.hpp>
#include <composyx/precond/AbstractSchwarz.hpp>
#include <composyx/precond/TwoLevelAbstractSchwarz.hpp>


#include <composyx/wrappers/Eigen/Eigen_header.hpp>
#include <composyx/wrappers/Eigen/Eigen.hpp>
#include <composyx/wrappers/Eigen/EigenSparseSolver.hpp>

#endif

BEGIN_COMPOSYXINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
struct ComposyxEnv
{
  static void initialize(Arccore::MessagePassing::IMessagePassingMng* parallel_mng);
  static void finalize();
  static ComposyxEnv* instance() {
    return m_instance ;
  }
  void init(IMessagePassingMng* parallel_mng)
  {
    m_parallel_mng = parallel_mng;
#ifdef ALIEN_USE_COMPOSYX
    m_p = composyx::bind_subdomains(m_parallel_mng->commSize()) ;
#endif
    alien_info([&] {
      cout() << "COMPOSYX ENV ::INIT : "<<m_parallel_mng->commSize();
    });
  }

#ifdef ALIEN_USE_COMPOSYX
  std::shared_ptr<composyx::Process> process() {
    return m_p ;
  }
#endif

  template <typename ValueT>
  static ValueT getEnv(std::string const& key, ValueT default_value);

 private:
  static bool m_is_initialized;
  static ComposyxEnv* m_instance ;
  IMessagePassingMng* m_parallel_mng = nullptr;
#ifdef ALIEN_USE_COMPOSYX
  std::shared_ptr<composyx::Process> m_p ;
#endif
};

template <typename ValueT>
class MatrixInternal
{
 public:
  typedef ValueT                  ValueType;
  typedef SimpleCSRMatrix<ValueT> CSRMatrixType;
  typedef SimpleCSRVector<ValueT> CSRVectorType;


#ifdef ALIEN_USE_COMPOSYX
  typedef composyx::SparseMatrixCSC<ValueT>    CSCMatrixType;
  typedef composyx::SparseMatrixCOO<ValueType> LocMatrixType ;
  typedef composyx::PartMatrix<LocMatrixType>  MatrixType ;
  //ComposyxMatrixType& matrix() { return m_matrix; }
#endif

  void init(IMessagePassingMng* parallel_mng)
  {
    m_parallel_mng = parallel_mng;
    m_nb_subdomains = m_parallel_mng->commSize() ;
    m_sd_id = m_parallel_mng->commRank() ;
    m_parallel = m_nb_subdomains>1 ;
    alien_info([&] {
      cout() << "INIT COMPOSYX : "<<m_nb_subdomains<<" "<<m_parallel;
    });
  }

  void compute(CSRMatrixType  const& matrix)
  {
    alien_info([&] {
      cout() << "CSR TO COMPOSYX : "<<m_nb_subdomains<<" "<<m_parallel;
    });


    auto const& matrix_profile = matrix.getCSRProfile();

    m_global_size = matrix.getGlobalSize() ;
    m_local_offset = matrix.getLocalOffset() ;
    m_n_dofs      = matrix.getLocalSize() ;
    m_nnz         = matrix_profile.getNnz() ;
    alien_info([&] {
      cout() << "N NDOF NNZ : "<<m_global_size<<" "<<m_n_dofs<<" "<<m_nnz;
    });

    std::map<int, std::vector<int>> nei_map ;
    std::map<int, std::vector<int>> nei_owner_map ;
    int local_size = m_n_dofs ;
    if(m_parallel)
    {
      m_n_dofs += matrix.getGhostSize() ;
      m_nnz    += matrix.getGhostSize() ;

      //Compute Neighb map
      auto const& dist_info = matrix.getDistStructInfo() ;
      auto const& recv_info = dist_info.m_recv_info;
      auto const& send_info = dist_info.m_send_info;
      alien_info([&] {
        cout() << "NUM OF RECV NEIGHB : "<<recv_info.m_num_neighbours<<" "<<recv_info.m_ids.size()<<" "<<recv_info.m_uids.size();
        cout() << "NUM OF SEND NEIGHB : "<<send_info.m_num_neighbours<<" "<<send_info.m_ids.size()<<" "<<send_info.m_uids.size();
      });
      for(int ineighb=0;ineighb<recv_info.m_first_upper_neighb;++ineighb)
      {
          int neighb_rank = recv_info.m_ranks[ineighb] ;
          //alien_info([&] {
          //  cout() << "RECV NEIGHB : "<<ineighb<<" "<<neighb_rank;
          //});
        auto& neighb_nodes = nei_map[neighb_rank] ;
        auto& neighb_owners = nei_owner_map[neighb_rank] ;
        for(int k=recv_info.m_ids_offset[ineighb];k<recv_info.m_ids_offset[ineighb+1];++k)
        {
          /*
          alien_info([&] {
              cout() << "          RECV VERTEX : "<<k<<" "<<recv_info.m_uids[k-recv_info.m_ids_offset[0]]<<" "<<neighb_rank;
          });*/
          neighb_nodes.push_back(k) ;
          neighb_owners.push_back(neighb_rank) ;
        }
      }
      for(int ineighb=0;ineighb<send_info.m_num_neighbours;++ineighb)
      {
        int neighb_rank = send_info.m_ranks[ineighb] ;
        //alien_info([&] {
        //  cout() << "SEND NEIGHB : "<<ineighb<<" "<<neighb_rank;
        //});
        auto& neighb_nodes = nei_map[neighb_rank] ;
        auto& neighb_owners = nei_owner_map[neighb_rank] ;
        for(int k=send_info.m_ids_offset[ineighb];k<send_info.m_ids_offset[ineighb+1];++k)
        {
          /*
          alien_info([&] {
            cout() << "          SEND VERTEX : "<<k<<" "<<m_local_offset+send_info.m_ids[k]<<" "<<m_sd_id;
          });*/
          neighb_nodes.push_back(send_info.m_ids[k]) ;
          neighb_owners.push_back(m_sd_id) ;
        }
      }
      for(int ineighb=recv_info.m_first_upper_neighb;ineighb<recv_info.m_num_neighbours;++ineighb)
      {
        int neighb_rank = recv_info.m_ranks[ineighb] ;
        //alien_info([&] {
        //  cout() << "RECV NEIGHB : "<<ineighb<<" "<<neighb_rank;
        //});
        auto& neighb_nodes = nei_map[neighb_rank] ;
        auto& neighb_owners = nei_owner_map[neighb_rank] ;
        for(int k=recv_info.m_ids_offset[ineighb];k<recv_info.m_ids_offset[ineighb+1];++k)
        {
          /*
          alien_info([&] {
              cout() << "          RECV VERTEX : "<<k<<" "<<recv_info.m_uids[k-recv_info.m_ids_offset[0]]<<" "<<neighb_rank;
          });*/
          neighb_nodes.push_back(k) ;
          neighb_owners.push_back(neighb_rank) ;
        }
      }
    }

#ifdef ALIEN_USE_COMPOSYX
    m_sd.emplace_back(m_sd_id, m_n_dofs, std::move(nei_map),nei_owner_map,false);
    auto p = ComposyxInternal::ComposyxEnv::instance()->process() ;
    p->load_subdomains(m_sd);
    /*
    alien_info([&] {
      p->display("PROCESS",cout().file()) ;
    });*/

    int              nrows          = matrix_profile.getNRow();
    int const*       kcol           = matrix_profile.kcol();
    ValueType const* values         = matrix.data() ;
    if(m_parallel)
    {
        auto const& dist_info = matrix.getDistStructInfo() ;
        auto const& cols = dist_info.m_cols;
        alien_info([&] {
          cout() << "PAR NNZ : "<<m_global_size<<" "<<nrows<<" "<<m_n_dofs<<" "<<m_nnz;
        });
        m_a_i.reserve(m_nnz) ;
        m_a_j.reserve(m_nnz) ;
        m_a_v.reserve(m_nnz) ;
        for(int irow=0;irow<nrows;++irow)
        {
          for(int k=kcol[irow];k<kcol[irow+1];++k)
          {
              m_a_i.push_back(irow) ;
              m_a_j.push_back(cols[k]) ;
              m_a_v.push_back(values[k]) ;
              /*
              alien_info([&] {
                cout() << "ADD ENTRY : "<<irow<<" "<<cols[k]<<" "<<values[k];
              });*/
          }
        }

        for(int irow=nrows;irow<nrows+matrix.getGhostSize();++irow)
        {
            m_a_i.push_back(irow) ;
            m_a_j.push_back(irow) ;
            m_a_v.push_back(1.) ;
            /*
            alien_info([&] {
              cout() << "ADD ENTRY : "<<irow<<" "<<irow<<" "<<0.;
            });*/
        }
        m_A_loc.reset(new LocMatrixType(m_n_dofs, m_n_dofs, m_nnz, m_a_i.data(), m_a_j.data(), m_a_v.data()));
    }
    else
    {
        alien_info([&] {
          cout() << "SEQ NNZ : "<<m_global_size<<" "<<m_n_dofs<<" "<<m_nnz;
        });
        int const* cols = matrix_profile.cols();
        m_a_i.reserve(m_nnz) ;
        m_a_j.reserve(m_nnz) ;
        m_a_v.reserve(m_nnz) ;
        for(int irow=0;irow<nrows;++irow)
        {
          for(int k=kcol[irow];k<kcol[irow+1];++k)
          {
              m_a_i.push_back(irow) ;
              m_a_j.push_back(cols[k]) ;
              m_a_v.push_back(values[k]) ;
              /*
              alien_info([&] {
                cout() << "ADD ENTRY : "<<irow<<" "<<cols[k]<<" "<<values[k];
              });*/

          }
        }
        m_A_loc.reset(new LocMatrixType(m_n_dofs, m_n_dofs, m_nnz, m_a_i.data(), m_a_j.data(), m_a_v.data()));
    }

    for(int k = 0; k < m_nb_subdomains; ++k)
      if(p->owns_subdomain(k))
        m_A_map[k] = *m_A_loc;

    alien_info([&] {
      cout() << "CREATE A ";
    });
    m_A.reset(new MatrixType(p, std::move(m_A_map)));
#endif
  }

  bool isParallel() const {
    return m_parallel ;
  }

#ifdef ALIEN_USE_COMPOSYX
  MatrixType&  getMatrix() {
    return *m_A;
  }

  MatrixType const&  getMatrix() const{
    return *m_A;
  }


  LocMatrixType&  getLocMatrix() {
    return *m_A_loc;
  }

  LocMatrixType const&  getLocMatrix() const{
    return *m_A_loc;
  }
#endif

 private:
  IMessagePassingMng* m_parallel_mng = nullptr;

  int m_sd_id         = 0 ;
  int m_nb_subdomains = 1 ;
  bool m_parallel     = false ;
  int m_local_offset  = 0 ;
  int m_global_size   = 0 ;
  int m_n_dofs        = 0 ;
  int m_nnz           = 0 ;
#ifdef ALIEN_USE_COMPOSYX
  // Define topology for each subdomain
  std::vector<composyx::Subdomain> m_sd;
  std::map<int, std::vector<int>> m_nei_map ;

  // Local matrices (here the same matrix on every subdomain)
  std::vector<int>       m_a_i;
  std::vector<int>       m_a_j;
  std::vector<ValueType> m_a_v;

  std::unique_ptr<LocMatrixType> m_A_loc;
  std::map<int,LocMatrixType>    m_A_map;
  std::unique_ptr<MatrixType>    m_A;

#endif
};

/*---------------------------------------------------------------------------*/

template <typename ValueT>
class VectorInternal
{
 public:
  typedef ValueT ValueType;
  typedef SimpleCSRVector<ValueT>             CSRVectorType;
  typedef composyx::Vector<ValueType>         LocVectorType ;
  typedef composyx::PartVector<LocVectorType> VectorType ;

  void init(IMessagePassingMng const* parallel_mng)
  {
    m_parallel_mng = parallel_mng;
    m_nb_subdomains = m_parallel_mng->commSize() ;
    m_sd_id = m_parallel_mng->commRank() ;
    m_parallel = m_nb_subdomains>1 ;
    alien_info([&] {
      cout() << "INIT COMPOSYX : "<<m_nb_subdomains<<" "<<m_parallel;
    });
  }

  bool compute(CSRVectorType  const& vector)
  {
    m_ndofs = vector.getAllocSize() ;
    alien_info([&] {
      cout() << "VECTOR CSR TO COMPOSYX : "<<m_nb_subdomains<<" "<<m_parallel<<" "<<m_ndofs;
    });

    std::vector<ValueT> v(m_ndofs) ;
    std::copy(vector.data(),vector.data()+m_ndofs,v.data()) ;
    /*
    alien_info([&] {
      for(int irow=0;irow<m_ndofs;++irow)
      {
        cout() << "X["<<irow<<"]="<<v[irow];
      }
    });*/

#ifdef ALIEN_USE_COMPOSYX
    m_b_loc.reset(new LocVectorType(m_ndofs,1,const_cast<ValueType*>(vector.data()))) ;
    auto p = ComposyxInternal::ComposyxEnv::instance()->process() ;
    if(p->owns_subdomain(m_sd_id))
      m_b_map[m_sd_id] = *m_b_loc;

    // Create the partitioned vector
    m_b.reset( new composyx::PartVector<composyx::Vector<ValueType>>(p, std::move(m_b_map)));
#endif
    return true ;
  }

  void getValues(std::size_t nrows, ValueType* values)
  {
    assert(nrows<=m_ndofs) ;
    for(std::size_t i=0;i<nrows;++i)
      values[i] = (*m_b_loc)[i] ;
  }

  bool init(IMessagePassingMng const* parallel_mng, std::size_t ndofs, LocVectorType&& rhs)
  {
    init(parallel_mng) ;
    m_ndofs = ndofs ;

    alien_info([&] {
      cout() << "LOCVECTOR TO COMPOSYX : "<<m_nb_subdomains<<" "<<m_parallel<<" "<<m_ndofs;
    });

#ifdef ALIEN_USE_COMPOSYX
    m_b_loc.reset(new LocVectorType(rhs)) ;
    auto p = ComposyxInternal::ComposyxEnv::instance()->process() ;
    if(p->owns_subdomain(m_sd_id))
      m_b_map[m_sd_id] = *m_b_loc;

    // Create the partitioned vector
    m_b.reset( new composyx::PartVector<composyx::Vector<ValueType>>(p, std::move(m_b_map)));
#endif
    return true ;
  }

  bool init(IMessagePassingMng const* parallel_mng, std::size_t ndofs, VectorType&& rhs)
  {
    init(parallel_mng) ;
    m_ndofs = ndofs ;

    alien_info([&] {
      cout() << "LOCVECTOR TO COMPOSYX : "<<m_nb_subdomains<<" "<<m_parallel<<" "<<m_ndofs;
    });

#ifdef ALIEN_USE_COMPOSYX
    //m_b_loc.reset(new LocVectorType(rhs)) ;
    //if(m_p->owns_subdomain(m_sd_id))
    //  m_b_map[m_sd_id] = *m_b_loc;

    // Create the partitioned vector
    m_b.reset( new composyx::PartVector<composyx::Vector<ValueType>>(rhs));
#endif
    return true ;
  }

  IMessagePassingMng const* parallelMng() const {
    return m_parallel_mng ;
  }

  std::size_t ndofs() const {
    return m_ndofs;
  }

  bool isParallel() const {
    return m_parallel ;
  }

#ifdef ALIEN_USE_COMPOSYX
  VectorType& getVector() {
    return *m_b ;
  }

  VectorType const& getVector() const {
    return *m_b ;
  }

  LocVectorType& getLocVector() {
    return *m_b_loc ;
  }

  LocVectorType const& getLocVector() const {
    return *m_b_loc ;
  }
#endif

private :
  IMessagePassingMng const* m_parallel_mng = nullptr;

  std::size_t m_ndofs         = 0;
  int         m_sd_id         = 0 ;
  int         m_nb_subdomains = 1 ;
  bool        m_parallel      = false ;
#ifdef ALIEN_USE_COMPOSYX
  std::map<int,LocVectorType>   m_b_map;

  // Create the partitioned vector
  std::unique_ptr<LocVectorType> m_b_loc ;
  std::unique_ptr<VectorType>    m_b;
#endif

};


template <typename ValueT>
class SolverInternal
{
public:
  typedef  ValueT ValueType ;
#ifdef ALIEN_USE_COMPOSYX
  typedef composyx::SparseMatrixCOO<ValueType>                     LocMatrixType ;
  typedef composyx::Vector<ValueType>                              LocVectorType ;

  typedef composyx::PartMatrix<LocMatrixType>                      ParMatrixType ;
  typedef composyx::PartVector<LocVectorType>                      ParVectorType ;

  typedef composyx::DiagonalPrecond<LocMatrixType,LocVectorType>   SeqDiagPrecondType ;
  typedef composyx::DiagonalPrecond<ParMatrixType,ParVectorType>   ParDiagPrecondType ;

  typedef composyx::Jacobi<LocMatrixType,LocVectorType>            SeqJacobiSolverType ;
  typedef composyx::Jacobi<ParMatrixType,ParVectorType>            ParJacobiSolverType ;

  typedef Eigen::SparseMatrix<ValueType>                           EigenMatrixtype ;
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, 1>              EigenVectorType ;
  typedef composyx::EigenSparseSolver<EigenMatrixtype,
                                      EigenVectorType>             EigenDirectSolverType ;
  typedef composyx::Pastix<LocMatrixType,LocVectorType>            PastixDirectSolverType ;
  typedef PastixDirectSolverType                                   DirectSolverType;

  typedef composyx::AbstractSchwarz<ParMatrixType,
                                    ParVectorType,
                                    DirectSolverType>              ParAScharzPrecondType ;

  typedef composyx::ConjugateGradient<LocMatrixType,LocVectorType> SeqCGSolverType ;
  typedef composyx::ConjugateGradient<ParMatrixType,ParVectorType> ParCGSolverType ;
  typedef composyx::ConjugateGradient<LocMatrixType,
                                      LocVectorType,
                                      SeqDiagPrecondType>          SeqDiagCGSolverType ;
  typedef composyx::ConjugateGradient<ParMatrixType,
                                      ParVectorType,
                                      ParDiagPrecondType>          ParDiagCGSolverType ;
  typedef composyx::ConjugateGradient<LocMatrixType,
                                      LocVectorType,
                                      SeqJacobiSolverType>         SeqASCGSolverType ;
  typedef composyx::ConjugateGradient<ParMatrixType,
                                      ParVectorType,
                                      ParAScharzPrecondType>       ParASCGSolverType ;

  typedef composyx::BiCGSTAB<LocMatrixType,LocVectorType>          SeqBCGSSolverType ;
  typedef composyx::BiCGSTAB<ParMatrixType,ParVectorType>          ParBCGSSolverType ;
  typedef composyx::BiCGSTAB<LocMatrixType,
                             LocVectorType,
                             SeqDiagPrecondType>                   SeqDiagBCGSSolverType ;
  typedef composyx::BiCGSTAB<ParMatrixType,
                             ParVectorType,
                             ParDiagPrecondType>                   ParDiagBCGSSolverType ;
  typedef composyx::BiCGSTAB<LocMatrixType,
                             LocVectorType,
                             SeqDiagPrecondType>                   SeqASBCGSSolverType ;
  typedef composyx::BiCGSTAB<ParMatrixType,
                             ParVectorType,
                             ParAScharzPrecondType>                ParASBCGSSolverType ;

  typedef composyx::GMRES<LocMatrixType,LocVectorType>             SeqGMRESSolverType ;
  typedef composyx::GMRES<ParMatrixType,ParVectorType>             ParGMRESSolverType ;
  typedef composyx::GMRES<LocMatrixType,
                          LocVectorType,
                          SeqDiagPrecondType>                      SeqDiagGMRESSolverType ;
  typedef composyx::GMRES<ParMatrixType,
                          ParVectorType,
                          ParDiagPrecondType>                      ParDiagGMRESSolverType ;
  typedef composyx::GMRES<LocMatrixType,
                          LocVectorType,
                          SeqDiagPrecondType>                      SeqASGMRESSolverType ;
  typedef composyx::GMRES<ParMatrixType,
                          ParVectorType,
                          ParAScharzPrecondType>                   ParASGMRESSolverType ;


  class IComposyxSolver
  {
  public:
    virtual ParVectorType solve(ParMatrixType const& A, ParVectorType const& b) = 0 ;
    virtual LocVectorType solve(LocMatrixType const& A, LocVectorType const& b) = 0 ;
    virtual int getNiter(bool is_parallel) const = 0 ;
  };

  template <typename ParSolverT, typename SeqSolverT>
  class ComposyxSolverT
        : public IComposyxSolver
  {
  public:
    template <typename... Types>
    void setup(const Types&... args) noexcept {
      m_par_solver.setup(args...) ;
      m_seq_solver.setup(args...) ;
    }

    ParVectorType solve(ParMatrixType const& A, ParVectorType const& b)
    {
      m_par_solver.setup(A);
      return m_par_solver * b;
    }

    LocVectorType solve(LocMatrixType const& A, LocVectorType const& b)
    {
      m_seq_solver.setup(A);
      return m_seq_solver * b;
    }

    int getNiter(bool is_parallel) const
    {
      if(is_parallel)
      {
        return m_par_solver.get_n_iter() ;
      }
      else
      {
        return m_seq_solver.get_n_iter() ;
      }
    }

  private:
    ParSolverT m_par_solver ;
    SeqSolverT m_seq_solver ;
  };

  void setParam(std::string const& key,int value)
  {
    m_int_param[key] = value ;
  }

  void setParam(std::string const& key,double value)
  {
    m_double_param[key] = value ;
  }

  void setParam(std::string const& key,std::string value)
  {
    m_str_param[key] = value ;
  }

  void init(int max_iter,
            double tol,
            std::string const& solver,
            std::string const& preconditioner,
            bool verbose)
  {
    using namespace composyx ;
    m_max_iter = max_iter ;
    m_tol      = tol;
    m_verbose  = verbose;
    if(solver.compare("cg")==0)
    {
        if(preconditioner.compare("none")==0)
        {
          auto* solver = new ComposyxSolverT<ParCGSolverType,SeqCGSolverType>{} ;
          m_solver.reset(solver) ;
          solver->setup(parameters::tolerance<Real>{tol},
                        parameters::max_iter<int>{max_iter},
                        parameters::verbose<bool>{verbose});
        }
        if(preconditioner.compare("diag")==0)
        {
          auto* solver = new ComposyxSolverT<ParDiagCGSolverType,SeqDiagCGSolverType>{} ;
          m_solver.reset(solver) ;
          solver->setup(parameters::tolerance<Real>{tol},
                        parameters::max_iter<int>{max_iter},
                        parameters::verbose<bool>{verbose});
        }
        if(preconditioner.compare("as")==0)
        {
          auto* solver = new ComposyxSolverT<ParASCGSolverType,SeqASCGSolverType>{} ;
          m_solver.reset(solver) ;
          solver->setup(parameters::tolerance<Real>{tol},
                        parameters::max_iter<int>{max_iter},
                        parameters::verbose<bool>{verbose});
        }
    }
    if(solver.compare("bcgs")==0)
    {
        if(preconditioner.compare("none")==0)
        {
          auto* solver = new ComposyxSolverT<ParBCGSSolverType,SeqBCGSSolverType>{} ;
          m_solver.reset(solver) ;
          solver->setup(parameters::tolerance<Real>{tol},
                        parameters::max_iter<int>{max_iter},
                        parameters::verbose<bool>{verbose});
        }
        if(preconditioner.compare("diag")==0)
        {
          auto* solver = new ComposyxSolverT<ParDiagBCGSSolverType,SeqDiagBCGSSolverType>{} ;
          m_solver.reset(solver) ;
          solver->setup(parameters::tolerance<Real>{tol},
                        parameters::max_iter<int>{max_iter},
                        parameters::verbose<bool>{verbose});
        }
        if(preconditioner.compare("as")==0)
        {
          auto* solver = new ComposyxSolverT<ParASBCGSSolverType,SeqASBCGSSolverType>{} ;
          m_solver.reset(solver) ;
          solver->setup(parameters::tolerance<Real>{tol},
                        parameters::max_iter<int>{max_iter},
                        parameters::verbose<bool>{verbose});
        }
    }
    if(solver.compare("gmres")==0)
    {
        int restart = m_int_param["gmres-restart"] ;
        double zeta = m_double_param["gmres-zeta"] ;
        const std::string ortho_str = m_str_param["gmres-ortho"] ;
        bool trace_be = false;
        Ortho ortho = Ortho::CGS;
        if(ortho_str == std::string("CGS")) { ortho = Ortho::CGS; }
        else if(ortho_str == std::string("CGS2")) { ortho = Ortho::CGS2; }
        else if(ortho_str == std::string("MGS")) { ortho = Ortho::MGS; }
        else if(ortho_str == std::string("MGS2")) { ortho = Ortho::MGS2; }
        if(preconditioner.compare("none")==0)
        {
            auto* solver = new ComposyxSolverT<ParGMRESSolverType,SeqGMRESSolverType>{} ;
            m_solver.reset(solver) ;

            solver->setup(parameters::tolerance<Real>{tol},
                          parameters::max_iter<int>{max_iter},
                          parameters::restart<int>{restart},
                          parameters::compression_accuracy<double>{zeta},
                          parameters::orthogonalization<Ortho>{ortho},
                          parameters::verbose_mem<bool>{true},
                          //parameters::always_true_residual<bool>{trace_be},
                          parameters::verbose<bool>{verbose});
        }
        if(preconditioner.compare("diag")==0)
        {
          auto* solver = new ComposyxSolverT<ParDiagGMRESSolverType,SeqDiagGMRESSolverType>{} ;
          m_solver.reset(solver) ;
          solver->setup(parameters::tolerance<Real>{tol},
                        parameters::max_iter<int>{max_iter},
                        parameters::restart<int>{restart},
                        parameters::compression_accuracy<double>{zeta},
                        parameters::orthogonalization<Ortho>{ortho},
                        parameters::verbose_mem<bool>{true},
                        parameters::always_true_residual<bool>{trace_be},
                        parameters::verbose<bool>{verbose});
        }
        if(preconditioner.compare("as")==0)
        {
          auto* solver = new ComposyxSolverT<ParASGMRESSolverType,SeqDiagGMRESSolverType>{} ;
          m_solver.reset(solver) ;
          solver->setup(parameters::tolerance<Real>{tol},
                        parameters::max_iter<int>{max_iter},
                        parameters::restart<int>{restart},
                        parameters::compression_accuracy<double>{zeta},
                        parameters::orthogonalization<Ortho>{ortho},
                        parameters::verbose_mem<bool>{true},
                        parameters::always_true_residual<bool>{trace_be},
                        parameters::verbose<bool>{verbose});
        }
    }
    m_verbose = verbose ;
    alien_info([&] {
      cout() << "COMPOSYX SOLVER : INIT";
      cout() << "              MAX ITER : "<<m_max_iter;
      cout() << "                   TOL : "<<m_tol;
      cout() << "                SOLVER : "<<solver;
      cout() << "               PRECOND : "<<preconditioner;
      cout() << "               VERBOSE : "<<m_verbose;
    });
  }

  ParVectorType solve(ParMatrixType const& A, ParVectorType const& b)
  {
    assert(m_solver.get() != nullptr) ;
    return m_solver->solve(A,b);
  }

  LocVectorType solve(LocMatrixType const& A, LocVectorType const& b)
  {
    assert(m_solver.get() != nullptr) ;
    return m_solver->solve(A,b);

#ifdef TEST_DEBUG
    auto xx = m_solver->solve(A,b);
    auto yy = A*xx ;
    auto rr = b - yy ;
    alien_info([&] {
      xx.display("XX=",cout().file()) ;
      yy.display("YY=",cout().file()) ;
      rr.display("RR=",cout().file()) ;
    });


    DirectSolverType invA ;
    invA.setup(A);
    auto x = invA*b ;
    auto y = A*x ;
    auto r = b - y ;
    alien_info([&] {
      b.display("B=",cout().file()) ;
      x.display("X=",cout().file()) ;
      y.display("Y=",cout().file()) ;
      r.display("R=",cout().file()) ;
    });
    using namespace composyx ;
    using Scalar = ValueType ;
    Eigen::SparseMatrix<Scalar> A_eigen;
    build_matrix<Scalar>(A_eigen, n_rows(A), n_cols(A), n_nonzero(A),
                         A.get_i_ptr(), A.get_j_ptr(), A.get_v_ptr(),
                         /*fill_symmetry*/ false);
    alien_info([&] {
      cout()<<"A EIGEN";
      cout() << A_eigen;
    });

    using E_Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    E_Vector b_e(n_rows(b));
    std::memcpy(get_ptr(b_e), get_ptr(b), sizeof(Scalar) * n_rows(b));
    MatrixProperties<Scalar> prop;
    EigenSparseSolver<Eigen::SparseMatrix<Scalar>, E_Vector> eigen_solver(A_eigen,prop);
    auto x_e = eigen_solver * b_e ;

    alien_info([&] {
      cout()<<"X EIGEN";
      for(int k = 0; k < n_rows(A); ++k){
        cout() << x_e[k];
      }
    });
    return xx;
#endif
  }

  int getNiter(bool is_parallel) const {
    assert(m_solver.get() != nullptr) ;
    return m_solver->getNiter(is_parallel) ;
  }
#endif

private:
#ifdef ALIEN_USE_COMPOSYX
  std::unique_ptr<IComposyxSolver> m_solver ;
#endif
  int    m_max_iter = 0 ;
  double m_tol      = 1e-12;
  bool   m_verbose  = false ;

  std::map<std::string,int>         m_int_param ;
  std::map<std::string,double>      m_double_param ;
  std::map<std::string,std::string> m_str_param ;

};
/*---------------------------------------------------------------------------*/

END_COMPOSYXINTERNAL_NAMESPACE
