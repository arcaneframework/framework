// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ICOUPLINGMODELSOLVER_H_
#define ICOUPLINGMODELSOLVER_H_

namespace Arcane {}
using namespace Arcane;

class INumericalModel;

//!Interface for coupling two SubDomainModel
class ICouplingModelSolver
{
public :
   virtual ~ICouplingModelSolver() {}
   virtual  bool solve(INumericalModel* model1, 
                       Integer seq1,
                       INumericalModel* model2, 
                       Integer seq2) = 0 ;
   virtual void setVerbose(bool flag) = 0 ;
} ;
#endif /*ICOUPLINGMODELSOLVER_H_*/
