// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef ALIEN_AMGXOPERATOR_DECL_HPP
#define ALIEN_AMGXOPERATOR_DECL_HPP

#if defined (HAVE_MUELU_AMGX)
#include <Teuchos_ParameterList.hpp>

#include <Tpetra_Operator.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Distributor.hpp>
#include <Tpetra_HashTable.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Import_Util.hpp>

#include "MueLu_Exceptions.hpp"
#include "MueLu_TimeMonitor.hpp"
#include "MueLu_TpetraOperator.hpp"
#include "MueLu_VerboseObject.hpp"

#include <cuda_runtime.h>
#include <amgx_c.h>

namespace MueLu {


  /*! @class AMGXOperator
      @ingroup MueLuAdapters
      @brief Adapter for AmgX library from Nvidia.

      This templated version of the class throws errors in all methods as AmgX is not implemented for datatypes where scalar!=double/float and ordinal !=int
 */
  template <class Scalar,
            class LocalOrdinal,
            class GlobalOrdinal,
            class Node>
  class ALIEN_AMGXOperator : 
   public TpetraOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node>, 
   public BaseClass {
  private:
    typedef Scalar          SC;
    typedef LocalOrdinal    LO;
    typedef GlobalOrdinal   GO;
    typedef Node            NO;

    typedef Tpetra::Map<LO,GO,NO>            Map;
    typedef Tpetra::MultiVector<SC,LO,GO,NO> MultiVector;

  public:

    //! @name Constructor/Destructor
    //@{

    //! Constructor
    ALIEN_AMGXOperator(const Teuchos::RCP<Tpetra::CrsMatrix<SC,LO,GO,NO> > &InA, Teuchos::ParameterList &paramListIn) { }

    //! Destructor.
    virtual ~ALIEN_AMGXOperator() {}

    //@}

    //! Returns the Tpetra::Map object associated with the domain of this operator.
    Teuchos::RCP<const Map> getDomainMap() const{
      throw Exceptions::RuntimeError("Cannot use AMGXOperator with scalar != double and/or global ordinal != int \n");
    }

    //! Returns the Tpetra::Map object associated with the range of this operator.
    Teuchos::RCP<const Map> getRangeMap() const{
      throw Exceptions::RuntimeError("Cannot use AMGXOperator with scalar != double and/or global ordinal != int \n");
    }

    //! Returns a solution for the linear system AX=Y in the  Tpetra::MultiVector X.
    /*!
      \param[in]  X - Tpetra::MultiVector of dimension NumVectors that contains the solution to the linear system.
      \param[out] Y -Tpetra::MultiVector of dimension NumVectors containing the RHS of the linear system.
    */
    void apply(const MultiVector& X, MultiVector& Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(), Scalar beta  = Teuchos::ScalarTraits<Scalar>::zero()) const {
      throw Exceptions::RuntimeError("Cannot use AMGXOperator with scalar != double and/or global ordinal != int \n");
    }

    //! Indicates whether this operator supports applying the adjoint operator
    bool hasTransposeApply() const{
      throw Exceptions::RuntimeError("Cannot use AMGXOperator with scalar != double and/or global ordinal != int \n");
    }

    RCP<MueLu::Hierarchy<SC,LO,GO,NO> > GetHierarchy() const {
      throw Exceptions::RuntimeError("AMGXOperator does not hold a MueLu::Hierarchy object \n");
    }

  private:
  };

  /*! @class AMGXOperator
      @ingroup MueLuAdapters
      @brief Adapter for AmgX library from Nvidia.

      Creates an AmgX Solver object with a Tpetra Matrix. Partial specialization of the template for data types supported by AmgX.
  */
  template<class Node>
  class ALIEN_AMGXOperator<double, int, int, Node> : public TpetraOperator<double, int, int, Node> {
  private:
    typedef double  SC;
    typedef int     LO;
    typedef int     GO;
    typedef Node    NO;

    typedef Tpetra::Map<LO,GO,NO>            Map;
    typedef Tpetra::MultiVector<SC,LO,GO,NO> MultiVector;

   
    void printMaps(Teuchos::RCP<const Teuchos::Comm<int> >& comm, const std::vector<std::vector<int> >& vec, const std::vector<int>& perm,
                   const int* nbrs, const Map& map, const std::string& label) {
      for (int p = 0; p < comm->getSize(); p++) {
        if (comm->getRank() == p) {
          std::cout << "========\n" << label << ", lid (gid), PID  " << p << "\n========" << std::endl;

          for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << "   neighbor " << nbrs[i] << " :";
            for (size_t j = 0; j < vec[i].size(); ++j)
              std::cout << " " << vec[i][j] << " (" << map.getGlobalElement(perm[vec[i][j]]) << ")";
            std::cout << std::endl;
          }
          std::cout << std::endl;
        } else {
          sleep(1);
        }
        comm->barrier();
      }
    }

  public:
    ALIEN_AMGXOperator(RCP<const Teuchos::Comm<int> > comm, Teuchos::ParameterList &paramListIn) {
          int numProcs = comm->getSize();
          int myRank   = comm->getRank();

          RCP<Teuchos::Time> amgxTimer = Teuchos::TimeMonitor::getNewTimer("MueLu: AMGX: initialize");
          amgxTimer->start();
	  Teuchos::ParameterList configs = paramListIn.sublist("amgx:params", true);
          if (configs.isParameter("json file")) {
            m_amgx_env.m_config_file = configs.get<std::string>("json file") ;
	    m_amgx_env.m_config_string =  "empty";
          } else {
	    m_amgx_env.m_config_file = "undefined" ;
            std::ostringstream oss;
            oss << "";
            ParameterList::ConstIterator itr;
            for (itr = configs.begin(); itr != configs.end(); ++itr) {
              const std::string&    name  = configs.name(itr);
              const ParameterEntry& entry = configs.entry(itr);
              oss << name << "=" << filterValueToString(entry) << ", ";
            }
            oss << "\0";
	    m_amgx_env.m_config_string = oss.str();
          }
	  Alien::TrilinosInternal::initAMGX(m_amgx_env) ;

          amgxTimer->stop();
          amgxTimer->incrementNumCalls();
    }


    //! Destructor.
    virtual ~ALIEN_AMGXOperator()
    {
      // Comment this out if you need rebuild to work. This causes AMGX_solver_destroy memory issues.
    }

    void init(const Teuchos::RCP<Tpetra::CrsMatrix<SC,LO,GO,NO> > &inA) 
    {
      domainMap_  = inA->getDomainMap();
      rangeMap_   = inA->getRangeMap();

      Alien::TrilinosInternal::initAMGX(m_amgx_env,*inA) ;

      vectorTimer1_ = Teuchos::TimeMonitor::getNewTimer("MueLu: AMGX: transfer vectors CPU->GPU");
      vectorTimer2_ = Teuchos::TimeMonitor::getNewTimer("MueLu: AMGX: transfer vector  GPU->CPU");
      solverTimer_  = Teuchos::TimeMonitor::getNewTimer("MueLu: AMGX: Solve (total)");
    }
   
    //! Returns the Tpetra::Map object associated with the domain of this operator.
    Teuchos::RCP<const Map> getDomainMap() const;

    //! Returns the Tpetra::Map object associated with the range of this operator.
    Teuchos::RCP<const Map> getRangeMap() const;

    //! Returns in X the solution to the linear system AX=Y.
    /*!
       \param[out] X - Tpetra::MultiVector of dimension NumVectors containing the RHS of the linear system
       \param[in]  Y - Tpetra::MultiVector of dimension NumVectors containing the solution to the linear system
       */
    void apply(const MultiVector& X, MultiVector& Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               SC alpha = Teuchos::ScalarTraits<SC>::one(), SC beta = Teuchos::ScalarTraits<SC>::zero()) const;

    //! Indicates whether this operator supports applying the adjoint operator.
    bool hasTransposeApply() const;

    RCP<MueLu::Hierarchy<SC,LO,GO,NO> > GetHierarchy() const {
      throw Exceptions::RuntimeError("AMGXOperator does not hold a MueLu::Hierarchy object \n");
    }

    std::string filterValueToString(const Teuchos::ParameterEntry& entry ) {
      return ( entry.isList() ? std::string("...") : toString(entry.getAny()) );
    }

    int sizeA() {
      int sizeX, sizeY, n;
      AMGX_matrix_get_size(m_amgx_env.m_A, &n, &sizeX, &sizeY);
      return n;
    }

    int iters() {
      int it;
      AMGX_solver_get_iterations_number(m_amgx_env.m_solver, &it);
      return it;
    }

    AMGX_SOLVE_STATUS getStatus() {
      AMGX_SOLVE_STATUS status;
      AMGX_solver_get_status(m_amgx_env.m_solver, &status);
      return status;
    }


  private:
    mutable Alien::TrilinosInternal::AMGXEnv m_amgx_env ;

    RCP<const Map>          domainMap_;
    RCP<const Map>          rangeMap_;

    RCP<Teuchos::Time>      vectorTimer1_;
    RCP<Teuchos::Time>      vectorTimer2_;
    RCP<Teuchos::Time>      solverTimer_;
  };

} // namespace

#endif //HAVE_MUELU_AMGX
#endif // ALIEN_AMGXOPERATOR_DECL_HPP
