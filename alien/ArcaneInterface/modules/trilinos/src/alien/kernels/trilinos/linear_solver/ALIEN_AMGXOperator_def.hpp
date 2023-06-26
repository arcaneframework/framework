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

#ifndef ALIEN_AMGXOPERATOR_DEF_HPP
#define ALIEN_AMGXOPERATOR_DEF_HPP

namespace Alien {
namespace TrilinosInternal {
void solveAMGX(Alien::TrilinosInternal::AMGXEnv& amgx_env, double* x, double* y) ;
}
}

#if defined (HAVE_MUELU_AMGX)
#include "MueLu_AMGXOperator_decl.hpp"

namespace MueLu {

  template<class Node>
  Teuchos::RCP<const Tpetra::Map<int,int,Node> >
  ALIEN_AMGXOperator<double,int,int,Node>::getDomainMap() const {
    return domainMap_;
  }

  template<class Node>
  Teuchos::RCP<const Tpetra::Map<int,int,Node> > ALIEN_AMGXOperator<double,int,int,Node>::getRangeMap() const {
    return rangeMap_;
  }

  template<class Node>
  void ALIEN_AMGXOperator<double,int,int,Node>::apply(const Tpetra::MultiVector<double,int,int,Node>& X,
                                                Tpetra::MultiVector<double,int,int,Node>&       Y,
                                                Teuchos::ETransp mode, double alpha, double beta) const {

    //std::cout<<"AMGX OPERATOR APPLY : "<<std::endl ;
    RCP<const Teuchos::Comm<int> > comm = Y.getMap()->getComm();
    
    ArrayRCP<const double> mueluXdata, amgxXdata;
    ArrayRCP<double>       mueluYdata, amgxYdata;

    try {
      for (int i = 0; i < (int)Y.getNumVectors(); i++) {
        {
          vectorTimer1_->start();

          mueluXdata = X.getData(i);
          mueluYdata = Y.getDataNonConst(i);

          if (comm->getSize() == 1) {
            //std::cout<<"COPY DATA FROM MUELU VECTOR"<<std::endl ;
            amgxXdata = mueluXdata;
            amgxYdata = mueluYdata;
            //std::cout<<"COPY DATA FROM MUELU VECTOR OK"<<std::endl ;
          } else {
            int n = mueluXdata.size();

            amgxXdata.resize(n);
            amgxYdata.resize(n);

            ArrayRCP<double> amgxXdata_nonConst = Teuchos::arcp_const_cast<double>(amgxXdata);
            for (int j = 0; j < n; j++) {
              amgxXdata_nonConst[m_amgx_env.m_muelu2amgx[j]] = mueluXdata[j];
              amgxYdata         [m_amgx_env.m_muelu2amgx[j]] = mueluYdata[j];
            }
          }

          vectorTimer1_->stop();
          vectorTimer1_->incrementNumCalls();
        }
       
        // Solve the system and time. 
        solverTimer_->start();
        if (comm->getSize() == 1) {
	  Alien::TrilinosInternal::solveAMGX(m_amgx_env,(double*)&mueluXdata[0],&mueluYdata[0]) ;
	} else {
	  Alien::TrilinosInternal::solveAMGX(m_amgx_env,(double*)&amgxXdata[0],&amgxYdata[0]) ;
	}
        solverTimer_->stop();
        solverTimer_->incrementNumCalls();

        {
          vectorTimer2_->start();

          if (comm->getSize() > 1) {
            int n = mueluYdata.size();

            for (int j = 0; j < n; j++)
              mueluYdata[j] = amgxYdata[m_amgx_env.m_muelu2amgx[j]];
          }

          vectorTimer2_->stop();
          vectorTimer2_->incrementNumCalls();
        }
      }

    } catch (std::exception& e) {
      std::string errMsg = std::string("Caught an exception in MueLu::AMGXOperator::Apply():\n") + e.what() + "\n";
      throw Exceptions::RuntimeError(errMsg);
    }
  }

  template<class Node>
  bool ALIEN_AMGXOperator<double,int,int,Node>::hasTransposeApply() const {
    return false;
  }

} // namespace
#endif //if defined(HAVE_MUELU_AMGX)

#endif //ifdef MUELU_AMGXOPERATOR_DEF_HPP
