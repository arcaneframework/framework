// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef LINEARSYATEMASSEMBLEOP
#define LINEARSYATEMASSEMBLEOP

#include "Numerics/LinearSolver/Impl/LinearSystemTwoStepBuilder.h"

//#define DEBUG_INFO
template<class CellExpr>
Real upStream(const Cell& T0,const Cell& T1,CellExpr& cell_exp)
{
  return 1. ;
}

template<typename SystemBuilder>
struct MassTerm
{
  template<typename MassExpr, typename RHSExpr>
  static void assemble(SystemBuilder& builder,
                        typename SystemBuilder::Entry const& uhi_entry,
                        typename SystemBuilder::Equation const& bal_eqn,
                        const CellGroup& internal_cells,
                        const MassExpr& mass,
                        const RHSExpr& rhs)
  {
    IIndexManager* manager = builder.getIndexManager();
    // on assemble le terme d/dt 
    ENUMERATE_CELL(icell, internal_cells)
    {  
      const Cell& cell = *icell ;
      if (icell->isOwn()) 
      {
        const IIndexManager::EquationIndex currentEquationIndex = manager->getEquationIndex(bal_eqn,cell);
        const IIndexManager::EntryIndex currentEntryIndex = manager->getEntryIndex(uhi_entry,cell);
        builder.addData(currentEquationIndex, currentEntryIndex, mass[cell] );
        // Vol/dt * P^n
        builder.setRHSData(currentEquationIndex,rhs[cell]);
//#define DEBUG_INFO
#ifdef DEBUG_INFO
        cout<<"MASS["<<currentEquationIndex<<"]["<<currentEntryIndex<<"]="<<mass[cell]<<" S="<<rhs[cell]<<endl;
#endif
      }
    }
  }
  
  static void defineProfile(LinearSystemTwoStepBuilder& builder,
                            LinearSystemTwoStepBuilder::Entry const& uhi_entry,
                            LinearSystemTwoStepBuilder::Equation const& bal_eqn,
                            const CellGroup& internal_cells)
  {
    IIndexManager* manager = builder.getIndexManager();
    // on assemble le terme d/dt 
    ENUMERATE_CELL(icell, internal_cells)
    {  
      const Cell & cell = *icell;
      builder.defineData(manager->defineEquationIndex(bal_eqn,cell),
                         manager->defineEntryIndex(uhi_entry,cell));
    }
  }
} ;

template<class SystemBuilder,class FluxModel>
struct FluxTerm
{
  template<typename LambdaExpr>
  static void assemble(SystemBuilder& builder,
                      typename SystemBuilder::Entry const& uhi_entry,
                      typename SystemBuilder::Equation const& bal_eqn,
                      const FluxModel* model,
                      const LambdaExpr& lambda_exp)
  {
    // DEFINE ENTRIES AND EQUATIONS  
    IIndexManager* index_manager = builder.getIndexManager();
    CoefficientArrayT<Cell>* cell_coefficients = model->getCellCoefficients() ;
  
    ////////////////////////////////////////////////////////////
    // Internal faces whose stencil contains cell unknowns only
    //
    ENUMERATE_FACE(iF, model->getCInternalFaces()) 
    {
      const Face& F = *iF;
  
      const Cell& T0  = F.backCell();  // back cell
      const Cell& T1  = F.frontCell(); // front cell
      Real lambda = upStream(T0,T1,lambda_exp) ;
  
      // May be improved using a global array
      Array<Integer> entry_indices(cell_coefficients->stencilSize(F));
      index_manager->getEntryIndex(uhi_entry,
                                   cell_coefficients->stencil(F),
                                   entry_indices);
  
      if(T0.isOwn())
      builder.addData(index_manager->getEquationIndex(bal_eqn, T0),
                             lambda,
                             entry_indices,
                             cell_coefficients->coefficients(F));
      if(T1.isOwn())
      builder.addData(index_manager->getEquationIndex(bal_eqn, T1),
                             -lambda,
                             entry_indices,
                             cell_coefficients->coefficients(F));
#ifdef DEBUG_INFO
      cout<<"M["<<index_manager->getEquationIndex(bal_eqn, T0)
             <<"]["<<entry_indices[0]<<"]="<<lambda*cell_coefficients->coefficients(F)[0]<<endl;
      cout<<"M["<<index_manager->getEquationIndex(bal_eqn, T0)
             <<"]["<<entry_indices[1]<<"]="<<lambda*cell_coefficients->coefficients(F)[1]<<endl;
      cout<<"M["<<index_manager->getEquationIndex(bal_eqn, T1)
             <<"]["<<entry_indices[0]<<"]="<<-lambda*cell_coefficients->coefficients(F)[0]<<endl;
      cout<<"M["<<index_manager->getEquationIndex(bal_eqn, T1)
             <<"]["<<entry_indices[1]<<"]="<<-lambda*cell_coefficients->coefficients(F)[1]<<endl;
#endif
    }
  }
  static void defineProfile(LinearSystemTwoStepBuilder& builder,
                            LinearSystemTwoStepBuilder::Entry const& uhi_entry,
                            LinearSystemTwoStepBuilder::Equation const& bal_eqn,
                            FluxModel const* model)
  {
    IIndexManager* index_manager = builder.getIndexManager();
    CoefficientArrayT<Cell>* cell_coefficients = model->getCellCoefficients() ;
      
    ////////////////////////////////////////////////////////////
    // Internal faces whose stencil contains cell unknowns only
    //
    ENUMERATE_FACE(iF, model->getCInternalFaces()) 
    {
      const Face& F = *iF;
  
      const Cell& T0  = F.backCell();  // back cell
      const Cell& T1  = F.frontCell(); // front cell
  
      // May be improved using a global array
      Integer stencil_size = cell_coefficients->stencilSize(F) ;
      ItemVectorView stencil = cell_coefficients->stencil(F) ;
      Array<Integer> entry_indices(stencil_size);
      for(Integer i=0;i<stencil_size;++i)
        entry_indices[i] = index_manager->defineEntryIndex(uhi_entry,stencil[i]) ;
  
      if(T0.isOwn())
        builder.defineData(index_manager->defineEquationIndex(bal_eqn, T0),
                          entry_indices) ;
      if(T1.isOwn())
        builder.defineData(index_manager->defineEquationIndex(bal_eqn, T1),
                         entry_indices) ;
    }
  }

  template<typename LambdaExpr>
  static void assemble(SystemBuilder& builder,
                      typename SystemBuilder::Entry const& uhi_entry,
                      typename SystemBuilder::Equation const& bal_eqn,
                      const LambdaExpr& velocity,
                      const FaceGroup& faces)
  {
    // DEFINE ENTRIES AND EQUATIONS  
    IIndexManager* index_manager = builder.getIndexManager();
  
    ////////////////////////////////////////////////////////////
    // Internal faces whose stencil contains cell unknowns only
    //
    ENUMERATE_FACE(iF, faces) 
    {
      const Face& F = *iF;
      const Cell& T0  = F.backCell();  // back cell
      const Cell& T1  = F.frontCell(); // front cell
      const IIndexManager::EntryIndex entryIndexT0 = index_manager->getEntryIndex(uhi_entry,T0);
      const IIndexManager::EntryIndex entryIndexT1 = index_manager->getEntryIndex(uhi_entry,T1);
      if(T0.isOwn())
        builder.addData(index_manager->getEquationIndex(bal_eqn, T0),
                        entryIndexT1,
                        velocity[F]) ;
      if(T1.isOwn())
        builder.addData(index_manager->getEquationIndex(bal_eqn, T1),
                        entryIndexT0,
                        -velocity[F]) ;
    }
  }
} ;

template<class SystemBuilder,class Model, typename Model::BCType bc_type>
struct BCIFluxTerm
{
  template<typename BCExpr>
  static
  void assemble(const Face& Fi,
                IIndexManager* index_manager,
                SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bal_eqn_T0,
                typename SystemBuilder::EquationIndex& bal_eqn_T1,
                typename SystemBuilder::Entry const& uhb_entry,
                Real taui,
                const BCExpr& bc_exp) ;
} ;

template<>
template< class SystemBuilder>
struct BCIFluxTerm<SystemBuilder,SubDomainModelProperty,SubDomainModelProperty::Neumann>
{
  template<typename BCExpr>
  static 
  void assemble(const Face& Fi,
                IIndexManager* index_manager,
                SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bal_eqn_T0,
                typename SystemBuilder::EquationIndex& bal_eqn_T1,
                typename SystemBuilder::Entry const& uhb_entry,
                Real taui,
                const BCExpr& bc_exp)
  {
    typename SystemBuilder::EntryIndex uhb_Fi = index_manager->getEntryIndex(uhb_entry, Fi);
    //if (T0.isOwn())
      builder.addData(bal_eqn_T0, uhb_Fi, taui);
    //if (T1.isOwn())
      builder.addData(bal_eqn_T1, uhb_Fi, -taui);
  }
} ;

template<>
template<class SystemBuilder>
struct BCIFluxTerm<SystemBuilder,SubDomainModelProperty,SubDomainModelProperty::Dirichlet>
{
  template<typename BCExpr>
  static
  void assemble(const Face& Fi,
                IIndexManager* index_manager,
                SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bal_eqn_T0,
                typename SystemBuilder::EquationIndex& bal_eqn_T1,
                typename SystemBuilder::Entry const& uhb_entry,
                Real taui,
                const BCExpr& bc_exp)
  {
   //if (T0.isOwn())
      builder.addRHSData(bal_eqn_T0, -taui * bc_exp[Fi]);
    //if (T1.isOwn())
      builder.addRHSData(bal_eqn_T1, taui * bc_exp[Fi]);
  }
} ;

template<class SystemBuilder,
         class FluxModel,
         class Model,
         typename Model::BCType bc_type>
struct MultiPtsBCIFluxTerm
{
  template<typename LambdaExpr,typename BCExpr>
  static 
  void assembleMultiPtsBCIFluxTerm(SystemBuilder& builder,
                                  typename SystemBuilder::Entry const& uhi_entry,
                                  typename SystemBuilder::Equation const& bal_eqn,
                                  typename SystemBuilder::Entry const& uhb_entry,
                                  typename SystemBuilder::Equation const& bnd_eqn,
                                  const FluxModel* model,
                                  const FaceGroup& cf_internal_faces,
                                  ItemGroupMapT<Face,typename Model::BCType>& face_bc_type,
                                  const LambdaExpr lambda_exp,
                                  const BCExpr& bc_exp)
  {
    ////////////////////////////////////////////////////////////
    // Internal faces whose stencil contains both cell and boundary face
    // unknowns
    CoefficientArrayT<Cell>* cell_coefficients = model->getCellCoefficients() ;
    CoefficientArrayT<Face>* face_coefficients = model->getFaceCoefficients();
    IIndexManager* index_manager = builder.getIndexManager();
    ENUMERATE_FACE(iF, cf_internal_faces) {
        const Face& F = *iF;
  
        const ItemVectorView& f_stencilF = face_coefficients->stencil(F);
  
        const Cell& T0  = F.backCell();  // back cell
        const Cell& T1  = F.frontCell(); // front cell
        Real lambda = upStream(lambda_exp[T0],lambda_exp[T1],lambda_exp) ;
        if (T0.isOwn() || T1.isOwn()) {
  
          typename SystemBuilder::EquationIndex bal_eqn_T0 = T0.isOwn() ? index_manager->getEquationIndex(bal_eqn, T0) : -1;
          typename SystemBuilder::EquationIndex bal_eqn_T1 = T1.isOwn() ? index_manager->getEquationIndex(bal_eqn, T1) : -1;
  
          // Cell coefficients
        // May be improved using a global array
        Array<Integer> entry_indices(cell_coefficients->stencilSize(F));
        index_manager->getEntryIndex(uhi_entry, cell_coefficients->stencil(F),
            entry_indices);
        if (T0.isOwn())
          builder.addData(index_manager->getEquationIndex(bal_eqn, T0), lambda,
              entry_indices, cell_coefficients->coefficients(F));
        if (T1.isOwn())
          builder.addData(index_manager->getEquationIndex(bal_eqn, T1), -lambda,
              entry_indices, cell_coefficients->coefficients(F));
  
          // Face coefficients
          ArrayView<Real> face_coefficients_F = face_coefficients->coefficients(F);
          int f_i = 0;
        for (ItemEnumerator Fi_it = f_stencilF.enumerator(); Fi_it.hasNext(); ++Fi_it)
        {
          const Face& Fi = (*Fi_it).toFace();
          Real taui = lambda*face_coefficients_F[f_i++];
          switch(face_bc_type[Fi]) 
          {
          case SubDomainModelProperty::Neumann :
            BCIFluxTerm<SystemBuilder,
                        Model,
                        SubDomainModelProperty::Neumann>::assemble(builder,
                                                                   index_manager,
                                                                   Fi,
                                                                   bal_eqn_T0,
                                                                   bal_eqn_T1,
                                                                   uhb_entry,
                                                                   taui,
                                                                   bc_exp) ;
            break ;
          case SubDomainModelProperty::Dirichlet :
            BCIFluxTerm<SystemBuilder,
                        Model,
                        SubDomainModelProperty::Dirichlet>::assemble(builder,
                                                                     index_manager,
                                                                     Fi,
                                                                     bal_eqn_T0,
                                                                     bal_eqn_T1,
                                                                     uhb_entry,
                                                                     taui,
                                                                     bc_exp) ;
            break ;
          }
        }
      }
    }
  }
} ;

template<class SystemBuilder,class Model, typename Model::BCType bc_type>
struct BCFluxTerm
{
  static
  void assemble(typename SystemBuilder::EquationIndex& bnd_eqn_F,
               typename SystemBuilder::EntryIndex& uhi_Ti1,
               Real taui) ;
} ;

template<>
template<class SystemBuilder>
struct BCFluxTerm<SystemBuilder,SubDomainModelProperty,SubDomainModelProperty::Neumann>
{
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                typename SystemBuilder::EntryIndex& uhi_Ti,
                Real taui)
  {
    builder.addData(bnd_eqn_F, uhi_Ti, taui) ;
  }
} ;

template<>
template<class SystemBuilder>
struct BCFluxTerm<SystemBuilder,SubDomainModelProperty,SubDomainModelProperty::Dirichlet>
{
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                typename SystemBuilder::EntryIndex& uhi_Ti1,
                Real taui)
  {
  }
} ;


template<class SystemBuilder, class Model, typename Model::BCType bc_type>
struct MassBCFluxTerm
{
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bal_eqn_T0,
                typename SystemBuilder::EntryIndex& uhb_Fi,
                Real taui,
                Real bc_Fi_value) ;
  static
  void assembleTwoPts(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bal_eqn_T0,
                typename SystemBuilder::EntryIndex& uhi_T0,
                Real taui,
                Real bc_Fi_value,
                Integer sgn) ;
} ;

template<>
template<class SystemBuilder>
struct MassBCFluxTerm<SystemBuilder,SubDomainModelProperty,SubDomainModelProperty::Neumann>
{
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bal_eqn_T0,
                typename SystemBuilder::EntryIndex& uhb_Fi,
                Real taui,
                Real bc_Fi_value)
  {
      builder.addData(bal_eqn_T0, uhb_Fi, taui);
  }
  static
  void assembleTwoPtsTest(SystemBuilder& builder,
      typename SystemBuilder::EquationIndex& bal_eqn_T0,
      typename SystemBuilder::EntryIndex& uhi_T0, Real taui, Real tausigma,
      Real bc_Fi_value, Integer sgn)
    {
#ifdef DEBUG_INFO
      cout<<"BOUNDARYFLUX NEUNMAN["<<bal_eqn_T0<<"]["<<uhi_T0<<"]"<<"  RHS="<<sgn<<"*"<<bc_Fi_value<<endl;
#endif
      builder.addRHSData(bal_eqn_T0, -bc_Fi_value);
    }
  static
  void assembleTwoPts(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bal_eqn_T0,
                typename SystemBuilder::EntryIndex& uhi_T0,
                Real taui,
                Real bc_Fi_value,
                Integer sgn)
  {
#ifdef DEBUG_INFO
    cout<<"BOUNDARYFLUX NEUNMAN["<<bal_eqn_T0<<"]["<<uhi_T0<<"]="<<taui/2<<"  RHS="<<sgn<<"*"<<bc_Fi_value<<endl;
#endif
    builder.addRHSData(bal_eqn_T0, sgn*bc_Fi_value);
  }
} ;
template<>
template<class SystemBuilder>
struct MassBCFluxTerm<SystemBuilder,SubDomainModelProperty,SubDomainModelProperty::Dirichlet>
{
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bal_eqn_T0,
                typename SystemBuilder::EntryIndex& uhb_Fi,
                Real taui,
                Real bc_Fi_value)
  {
      builder.addRHSData(bal_eqn_T0, -taui * bc_Fi_value);
  }
  static
  void assembleTwoPts(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bal_eqn_T0,
                typename SystemBuilder::EntryIndex& uhi_T0,
                Real taui,
                Real bc_Fi_value,
                Integer sgn)
  {
    builder.addData(bal_eqn_T0,uhi_T0,taui);
    builder.addRHSData(bal_eqn_T0, taui*bc_Fi_value);
#ifdef DEBUG_INFO
    cout<<"BOUNDARYFLUX DIRICHLET["<<bal_eqn_T0<<"]["<<uhi_T0<<"]="<<taui/2<<"  RHS="<<taui*bc_Fi_value<<endl;
#endif
  }
  static
  void assembleTwoPtsTest(SystemBuilder& builder,
      typename SystemBuilder::EquationIndex& bal_eqn_T0,
      typename SystemBuilder::EntryIndex& uhi_T0, Real taui, Real tausigma,
      Real bc_Fi_value, Integer sgn)
    {
      builder.addData(bal_eqn_T0, uhi_T0, sgn * taui);
      builder.addRHSData(bal_eqn_T0, -sgn * tausigma * bc_Fi_value);
#ifdef DEBUG_INFO
      cout<<"TEST BOUNDARYFLUX DIRICHLET["<<bal_eqn_T0<<"]["<<uhi_T0<<"]="<<sgn * taui<<"  RHS="<<sgn * tausigma<< "*" <<bc_Fi_value<<endl;
#endif
    }

  };

template< class SystemBuilder,
          class Model,
          typename Model::BCType bc_type>
struct BNDEqTerm
{
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                typename SystemBuilder::EntryIndex& uhb_Fi,
                Real taui,
                Real bc_Fi_value) ;
  template<typename BNDExpr,typename MeasureExpr>
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                const Face& F,
                const MeasureExpr& f_measures,
                const BNDExpr& bnd_exp,
                Real bc_F_value) ;
} ;

template<>
template<class SystemBuilder>
struct BNDEqTerm<SystemBuilder,SubDomainModelProperty,SubDomainModelProperty::Neumann>
{
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                typename SystemBuilder::EntryIndex& uhb_Fi,
                Real taui,
                Real bc_Fi_value)
  {
    builder.addData(bnd_eqn_F, uhb_Fi, taui);
  }
  template<typename BNDExpr,typename MeasureExpr>
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                const Face& F,
                const MeasureExpr& f_measures,
                const BNDExpr& bnd_exp,
                Real bc_F_value)
  {
    Real meas_F = f_measures[F];
    builder.addRHSData(bnd_eqn_F, meas_F * bc_F_value);
  }
} ;
template<>
template<class SystemBuilder>
struct BNDEqTerm<SystemBuilder,SubDomainModelProperty,SubDomainModelProperty::Dirichlet>
{
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                typename SystemBuilder::EntryIndex& uhb_Fi,
                Real taui,
                Real bc_Fi_value)
  {
    builder.addRHSData(bnd_eqn_F, -taui * bc_Fi_value);
  }
  template<typename BNDExpr,typename MeasureExpr>
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                const Face& F,
                const MeasureExpr& f_measures,
                const BNDExpr& bnd_exp,
                Real bc_F_value)
  {
    ;
  }
} ;


template< class SystemBuilder, 
          class Model1,
          typename Model1::BCType bc_type1,
          class Model2,
          typename Model2::BCType bc_type2>
struct BNDEqBCTerm
{
  template<typename BNDExpr>
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                const Face& F,
                const BNDExpr& bnd_exp,
                Real bc_F_value) ;
} ;

template<>
template<class SystemBuilder, class Model1, typename Model1::BCType bc_type1>
struct BNDEqBCTerm<SystemBuilder,Model1,bc_type1,SubDomainModelProperty,SubDomainModelProperty::Neumann>
{
  template<typename BNDExpr>
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                const Face& F,
                const BNDExpr& bnd_exp,
                Real bc_F_value)
  {
    builder.addRHSData(bnd_eqn_F, bnd_exp[F] * bc_F_value);
  }
} ;

template<>
template<class SystemBuilder,class Model1, typename Model1::BCType bc_type1>
struct BNDEqBCTerm<SystemBuilder,Model1,bc_type1,SubDomainModelProperty,SubDomainModelProperty::Dirichlet>
{
  template<typename BNDExpr>
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::EquationIndex& bnd_eqn_F,
                const Face& F,
                const BNDExpr& bnd_exp,
                Real bc_F_value)
  {
  }
} ;

//typedef SubDomainModelProperty::eBoundaryConditionType BCType ;
template< class SystemBuilder,
          class FluxModel,
          class Model,
          typename Model::BCType bc_type>
struct TwoPtsBCFluxTerm
{
  template<typename LambdaExpr,typename BCExpr>
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::Entry const& uhi_entry,
                typename SystemBuilder::Equation const& bal_eqn,
                typename SystemBuilder::Entry const& uhb_entry,
                const FluxModel* model,
                const FaceGroup& boundary_faces,
                const ItemGroupMapT<Face,Cell>& boundary_cells,
                const ItemGroupMapT<Face,Integer>& boundary_sgn,
                const LambdaExpr& lambda_exp,
                const BCExpr& bc_exp)
  {
    ////////////////////////////////////////////////////////////
    // Boundary faces
    CoefficientArrayT<Cell>* cell_coefficients = model->getCellCoefficients();
    CoefficientArrayT<Face>* face_coefficients = model->getFaceCoefficients();
    IIndexManager* index_manager = builder.getIndexManager();
    ENUMERATE_FACE(iF, boundary_faces.own()) 
    {
        const Face& F = *iF;
  
        ItemVectorView c_stencilF = cell_coefficients->stencil(F);
        ItemVectorView f_stencilF = face_coefficients->stencil(F);
  
        //const Cell& T0 = F.boundaryCell(); // boundary cell
        const Cell& T0 = boundary_cells[F] ;
        

#ifdef DEV
        Real lambda = lambda_exp[T0] ;
#endif

        // Element balance
        typename SystemBuilder::EquationIndex bal_eqn_T0 = index_manager->getEquationIndex(bal_eqn, T0);
        typename SystemBuilder::EntryIndex uhi_T0  = index_manager->getEntryIndex(uhi_entry, T0);
            ConstArrayView<Real> cell_coefficients_F = cell_coefficients->coefficients(F);
            ConstArrayView<Real> face_coefficients_F = face_coefficients->coefficients(F);

#warning PATCH FOR OVERLAP DIRICHLET CONDITION TO BE FIXED
            Real tau_sigma = 0.;
            Real tau_i = 0.;
            // Case of Overlap Dirichlet condition
            if(face_coefficients_F.size()==0 && cell_coefficients_F.size()==2)
              {
                int c_i = 0;
                for(ItemEnumerator Ti_it = c_stencilF.enumerator(); Ti_it.hasNext(); ++Ti_it)
                  {
                    const Cell& Ti = (*Ti_it).toCell();
                    if (Ti == T0)
                      tau_i = cell_coefficients_F[c_i++];
                    else
                      tau_sigma = cell_coefficients_F[c_i++];
                  }
              }

            // Case of REAL Dirichlet condition on global boundary

            else if(face_coefficients_F.size()==1 && cell_coefficients_F.size()==1)
              {
                tau_i = cell_coefficients_F[0];
                tau_sigma = face_coefficients_F[0];
              }

            // Wrong case

            else
            std::cout << "FATAL : Inconsistent Stencil Sizes, Do not know what to do!";

            MassBCFluxTerm<SystemBuilder,Model,bc_type>::assembleTwoPtsTest(builder,
                                                                    bal_eqn_T0,
                                                                    uhi_T0,
                                                                    tau_i,
                                                                    tau_sigma,
                                                                    bc_exp[F],
                                                                    boundary_sgn[F]) ;
    }
  }
} ;

template< class SystemBuilder,
          class FluxModel,
          class Model,
          typename Model::BCType bc_type>
struct MultiPtsBCFluxTerm
{
  template<typename LambdaExpr,typename BCExpr>
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::Entry const& uhi_entry,
                typename SystemBuilder::Equation const& bal_eqn,
                typename SystemBuilder::Entry const& uhb_entry,
                typename SystemBuilder::Equation const& bnd_eqn,
                const FluxModel* model,
                const FaceGroup& boundary_faces,
                const LambdaExpr& lambda_exp,
                const BCExpr& bc_exp)
  {
    ////////////////////////////////////////////////////////////
    // Boundary faces
    CoefficientArrayT<Cell>* cell_coefficients = model->getCellCoefficients();
    CoefficientArrayT<Face>* face_coefficients = model->getFaceCoefficients();
    IIndexManager* index_manager = builder.getIndexManager();
    const ItemGroupMapT<Face,Cell>& boundary_cells = model->boundaryCells() ;
    ENUMERATE_FACE(iF, boundary_faces.own()) 
    {
        const Face& F = *iF;
  
        ItemVectorView c_stencilF = cell_coefficients->stencil(F);
        ItemVectorView f_stencilF = face_coefficients->stencil(F);
  
        //const Cell& T0 = F.boundaryCell(); // boundary cell
        const Cell& T0 = boundary_cells[F] ;
        
        Real lambda = lambda_exp[T0] ;
  
        //if(!T0.isOwn()) fatal() << "Boundary cell expected to be own";
  
        // Element balance
        typename SystemBuilder::EquationIndex bal_eqn_T0 = index_manager->getEquationIndex(bal_eqn, T0);
  
        Real bc_F_value = bc_exp[F];
  
        ConstArrayView<Real> cell_coefficients_F = cell_coefficients->coefficients(F);
        ConstArrayView<Real> face_coefficients_F = face_coefficients->coefficients(F);
  
        // Cell coefficients
        int c_i = 0;
        for(ItemEnumerator Ti_it = c_stencilF.enumerator();  Ti_it.hasNext(); ++Ti_it) 
        {
          const Cell& Ti = (*Ti_it).toCell();
          Real taui = cell_coefficients_F[c_i++];
  
          typename SystemBuilder::EntryIndex uhi_Ti  = index_manager->getEntryIndex(uhi_entry, Ti);
  
          // Balance equation
          builder.addData(bal_eqn_T0, uhi_Ti, lambda*taui);
        }
  
        // Face coefficients
        int f_i = 0;
        for(ItemEnumerator Fi_it = f_stencilF.enumerator(); Fi_it.hasNext(); ++Fi_it) 
        {
          const Face& Fi = (*Fi_it).toFace();
          Real taui = face_coefficients_F[f_i++];
          typename SystemBuilder::EntryIndex uhb_Fi = index_manager->getEntryIndex(uhb_entry, Fi);
          MassBCFluxTerm<SystemBuilder,Model,bc_type>::assemble(builder,bal_eqn_T0,uhb_Fi,taui,bc_exp[Fi]) ;
        }
    }
  }
} ;


template< class SystemBuilder,
          class FluxModel, 
          class Model,
          typename Model::BCType bc_type>
struct MultiPtsBNDTerm
{
  template<typename LambdaExpr, 
           typename BCExpr, 
           typename BNDExpr, 
           typename MeasureExpr>
  static
  void assemble(SystemBuilder& builder,
                typename SystemBuilder::Entry const& uhi_entry,
                typename SystemBuilder::Equation const& bal_eqn,
                typename SystemBuilder::Entry const& uhb_entry,
                typename SystemBuilder::Equation const& bnd_eqn,
                const FluxModel* model,
                const FaceGroup& boundary_faces,
                ItemGroupMapT<Face,typename Model::BCType>& face_bc_type,
                const MeasureExpr& face_measures,
                const LambdaExpr& lambda_exp,
                const BCExpr& bc_exp,
                const BNDExpr& bnd_exp)
  {
    ////////////////////////////////////////////////////////////
    // Boundary faces
    CoefficientArrayT<Cell>* cell_coefficients = model->getCellCoefficients();
    CoefficientArrayT<Face>* face_coefficients = model->getFaceCoefficients();
    IIndexManager* index_manager = builder.getIndexManager();
    const ItemGroupMapT<Face,Cell>& boundary_cells = model->boundaryCells() ;
    ENUMERATE_FACE(iF, boundary_faces.own()) 
    {
        const Face& F = *iF;
        ItemVectorView c_stencilF = cell_coefficients->stencil(F);
        ItemVectorView f_stencilF = face_coefficients->stencil(F);
  
        //const Cell& T0 = F.boundaryCell(); // boundary cell
        const Cell& T0 = boundary_cells[F] ;
        Real lambda = lambda_exp[T0] ;
  
        ConstArrayView<Real> cell_coefficients_F = cell_coefficients->coefficients(F);
        ConstArrayView<Real> face_coefficients_F = face_coefficients->coefficients(F);
  
        typename SystemBuilder::EquationIndex bnd_eqn_F = index_manager->getEquationIndex(bnd_eqn, F);
        int c_i = 0;
        for(ItemEnumerator Ti_it = c_stencilF.enumerator();  Ti_it.hasNext(); ++Ti_it) 
        {
          const Cell& Ti = (*Ti_it).toCell();
          Real taui = cell_coefficients_F[c_i++];
  
          typename SystemBuilder::EntryIndex uhi_Ti  = index_manager->getEntryIndex(uhi_entry, Ti);
          //typename SystemBuilder::EntryIndex uhi_Ti       = index_manager->getEntryIndex(uhi_entry, Ti);
          BNDEqTerm<SystemBuilder,Model,bc_type>::assemble(builder,bnd_eqn_F,uhi_Ti,taui) ;
        }
        // Face coefficients
        int f_i = 0;
        for(ItemEnumerator Fi_it = f_stencilF.enumerator(); Fi_it.hasNext(); ++Fi_it) 
        {
          const Face& Fi = (*Fi_it).toFace();
          Real taui = face_coefficients_F[f_i++];
          Real bc_Fi_value = bc_exp[Fi];
          typename SystemBuilder::EntryIndex uhb_Fi = index_manager->getEntryIndex(uhb_entry, Fi);
          switch(face_bc_type[Fi])
          {
          case SubDomainModelProperty::Dirichlet :
            BNDEqBCTerm<SystemBuilder,Model,bc_type,SubDomainModelProperty,SubDomainModelProperty::Dirichlet>::assemble(builder,bnd_eqn_F,uhb_Fi,taui,bc_exp[Fi]) ;
            break ;
          case SubDomainModelProperty::Neumann :
            BNDEqBCTerm<SystemBuilder,Model,bc_type,SubDomainModelProperty,SubDomainModelProperty::Neumann>::assemble(builder,bnd_eqn_F,uhb_Fi,taui,bc_exp[Fi]) ;
            break ;
          }
        }
        BNDEqTerm<SystemBuilder,Model,bc_type>::assemble(builder,bnd_eqn_F,F,face_measures,bnd_exp,bc_exp[F]) ;
    }
  }
} ;

#endif

