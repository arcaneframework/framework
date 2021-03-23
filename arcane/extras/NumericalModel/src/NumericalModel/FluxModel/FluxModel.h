// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef FLUXMODEL_H_
#define FLUXMODEL_H_

#include "NumericalModel/SubDomainModel/INumericalDomain.h"
#include "NumericalModel/SubDomainModel/NumericalDomain/NumericalDomainImpl.h"
#include "NumericalModel/Models/ISubDomainModel.h"

#include "Numerics/DiscreteOperator/CoefficientArray.h"

#include <boost/shared_ptr.hpp>

using namespace Arcane;

class IDivKGradDiscreteOperator;
class IGeometryMng;

class FluxModel : INumericalDomainModel
{
public :
  FluxModel(ISubDomainModel::NumericalDomain* domain,
            IDivKGradDiscreteOperator* scheme,
            IGeometryMng* geometry)
  : m_numerical_domain(domain)
  , m_scheme(scheme)
  , m_geometry(geometry)
  , m_is_initialized(false)
  {}
  virtual ~FluxModel(){}
  void init() { m_is_initialized = true ; }

  void setNumericalDomain(const INumericalDomain* domain)
  {
    m_numerical_domain = dynamic_cast<const ISubDomainModel::NumericalDomain*>(domain) ;
  }
  void start(const VariableCellReal3x3& cell_perm_k) ;

  template<typename Collection1,typename Collection2>
  void computeFlux(Collection1& flux,const Collection2& p, const FaceGroup& faces) ;

  template<typename Collection1,typename Collection2>
  void computeInternalFlux(Collection1& flux,const Collection2& p) ;

  template<typename Collection1,typename Collection2,typename LambdaT>
  void computeInternalFlux(Collection1& flux,
                           const Collection2& p,
                           const LambdaT& lambda_exp) ;

  CoefficientArrayT<Cell>* getCellCoefficients() const
  {
    return m_cell_coefficients.get() ;
  }
  CoefficientArrayT<Face>* getFaceCoefficients() const
  {
    return m_face_coefficients.get() ;
  }
  const FaceGroup& getCInternalFaces() const
  {
    return m_c_internal_faces ;
  }
  const FaceGroup& getCFInternalFaces() const
  {
    return m_c_internal_faces ;
  }
  const FaceGroup& getAllFaces() const
  {
    return m_faces ;
  }

  const ItemGroupMapT<Face,Cell>& getBoundaryCells() const
  {
    return m_boundary_cells ;
  }
  const ItemGroupMapT<Face,Integer>& getBoundarySgn() const
  {
    return m_boundary_sgn ;
  }

private :

  template<typename LambdaT, typename ExprT>
  Real upStream(const Cell& T0,
                const Cell& T1,
                const LambdaT& lambda,
                const ExprT& p)
  {
    return (p[T0]>p[T1]?lambda[T0]:lambda[T1]) ;
  }
  const ISubDomainModel::NumericalDomain* m_numerical_domain ;
  IDivKGradDiscreteOperator* m_scheme ;
  IGeometryMng* m_geometry ;

  //! Internal faces whose stencil contains cell unknowns only
  FaceGroup m_c_internal_faces;

  //! Internal faces whose stencil contains both cell and face unknowns
  FaceGroup m_cf_internal_faces;

  //! The group of all the faces
  FaceGroup m_faces;

  //! Cell coefficients
  boost::shared_ptr<CoefficientArrayT<Cell> > m_cell_coefficients;
  //! Face coefficients
  boost::shared_ptr<CoefficientArrayT<Face> > m_face_coefficients;

  ItemGroupMapT<Face,Cell> m_boundary_cells ;
  ItemGroupMapT<Face,Integer> m_boundary_sgn ;

  bool m_is_initialized ;
};


template<typename Collection1,typename Collection2>
void
FluxModel::
computeInternalFlux(Collection1& flux,const Collection2& p)
{
  CoefficientArrayT<Cell>* cell_coefficients = m_cell_coefficients.get() ;
  ENUMERATE_FACE(iface,m_c_internal_faces)
  {
    const Face& face = *iface ;
    ConstArrayView<Real> cell_coefficients_F = cell_coefficients->coefficients(face);
    ItemVectorView c_stencilF = cell_coefficients->stencil(face);
    // Cell coefficients
    int c_i = 0;
    Real local_flux = 0 ;
    for(ItemEnumerator Ti_it = c_stencilF.enumerator();  Ti_it.hasNext(); ++Ti_it)
    {
      const Cell& Ti = (*Ti_it).toCell();
      local_flux += cell_coefficients_F[c_i++]*p[Ti];
    }
    flux[iface] = local_flux ;
  }
}

template<typename Collection1,typename Collection2>
void
FluxModel::
computeFlux(Collection1& flux,const Collection2& p, const FaceGroup& faces)
{
  CoefficientArrayT<Cell>* cell_coefficients = m_cell_coefficients.get() ;
  ENUMERATE_FACE(iface,faces)
  {
    const Face& face = *iface ;
    ConstArrayView<Real> cell_coefficients_F = cell_coefficients->coefficients(face);
    ItemVectorView c_stencilF = cell_coefficients->stencil(face);
    // Cell coefficients
    int c_i = 0;
    Real local_flux = 0 ;
    for(ItemEnumerator Ti_it = c_stencilF.enumerator();  Ti_it.hasNext(); ++Ti_it)
    {
      const Cell& Ti = (*Ti_it).toCell();
      local_flux += cell_coefficients_F[c_i++]*p[Ti];
    }
    flux[iface] = local_flux ;
  }
}
template<typename Collection1,typename Collection2, typename LambdaT>
void
FluxModel::
computeInternalFlux(Collection1& flux,
                    const Collection2& p,
                    const LambdaT& lambda_exp)
{
  CoefficientArrayT<Cell>* cell_coefficients = m_cell_coefficients.get() ;
  ENUMERATE_FACE(iface,m_c_internal_faces)
  {
    const Face& face = *iface ;
    const Cell& T0  = face.backCell();  // back cell
    const Cell& T1  = face.frontCell(); // front cell
    Real lambda = upStream(T0,T1,lambda_exp,p) ;
    ConstArrayView<Real> cell_coefficients_F = cell_coefficients->coefficients(face);
    ItemVectorView c_stencilF = cell_coefficients->stencil(face);
    // Cell coefficients
    int c_i = 0;
    Real local_flux = 0 ;
    for(ItemEnumerator Ti_it = c_stencilF.enumerator();  Ti_it.hasNext(); ++Ti_it)
    {
      const Cell& Ti = (*Ti_it).toCell();
      local_flux += cell_coefficients_F[c_i++]*p[Ti];
    }
    flux[iface] = lambda*local_flux ;
  }
}
#endif /*FLUXMODEL_H_*/
