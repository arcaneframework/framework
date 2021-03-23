// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef FLUXMODELT_H_
#define FLUXMODELT_H_

#include "Mesh/Geometry/IGeometryMng.h"
#include "NumericalModel/SubDomainModel/INumericalDomain.h"
#include "NumericalModel/SubDomainModel/NumericalDomain/NumericalDomainImpl.h"
#include "NumericalModel/Models/ISubDomainModel.h"

#include "Numerics/DiscreteOperator/CoefficientArray.h"

#include <boost/shared_ptr.hpp>

#include "Numerics/DiscreteOperator/IDivKGradTwoPoints.h"

using namespace Arcane;

class IGeometryMng;

template<typename SchemeType>
class FluxModelT : INumericalDomainModel
{
public :
  FluxModelT(ISubDomainModel::NumericalDomain* domain,
            SchemeType* scheme,
            IGeometryMng* geometry)
  : m_numerical_domain(domain)
  , m_scheme(scheme)
  , m_geometry(geometry)
  , m_is_initialized(false)
  {}
  virtual ~FluxModelT(){}
  void init() { m_is_initialized = true ; }

  void setNumericalDomain(const INumericalDomain* domain)
  {
    m_numerical_domain = dynamic_cast<const ISubDomainModel::NumericalDomain*>(domain) ;
  }
  template<typename TensorType>
  void start(const TensorType& k) {
    const CellGroup& internal_cells = m_numerical_domain->internalCells() ;
    const FaceGroup& internal_faces = m_numerical_domain->internalFaces() ;

    const FaceGroup& boundary_faces = m_numerical_domain->boundaryFaces() ;
    // Build the group of all the faces
    ItemGroupBuilder<Face> faces_builder(internal_faces.mesh(), IMPLICIT_NAME);
    ENUMERATE_FACE(iF, internal_faces) {
      const Face& F = *iF;
      faces_builder.add(F);
    }
    m_boundary_cells.init(boundary_faces) ;
    m_boundary_sgn.init(boundary_faces) ;
    ItemGroupSet is_internal_cell ;
    is_internal_cell.init(internal_cells) ;
    ENUMERATE_FACE(iF, boundary_faces) {
      const Face& F = *iF;
      
      faces_builder.add(F);
      if(F.isBoundary())
      {
        m_boundary_cells[F] = F.boundaryCell() ;
        m_boundary_sgn[F] = 1 ; ;
      }
      else
      {
        if(is_internal_cell.hasKey(F.backCell()))
        {
          m_boundary_cells[F] = F.backCell() ;
          m_boundary_sgn[F] = 1 ;
        }
        else
        {
          m_boundary_cells[F] = F.frontCell() ;
          m_boundary_sgn[F] = -1 ; 
        }
      }
    }
    m_faces = faces_builder.buildGroup();
    
    m_cell_coefficients.reset(new CoefficientArrayT<Cell>(m_faces, internal_cells));
    m_face_coefficients.reset(new CoefficientArrayT<Face>(m_faces, boundary_faces));
    
    m_c_internal_faces  = internal_faces.itemFamily()->createGroup(IMPLICIT_UNIQ_NAME);
    m_cf_internal_faces = internal_faces.itemFamily()->createGroup(IMPLICIT_UNIQ_NAME);

    IDivKGradTwoPoints * two_points_scheme = dynamic_cast<IDivKGradTwoPoints *>(m_scheme);
    if(two_points_scheme) {
      two_points_scheme->setProperty(DiscreteOperatorProperty::P_CELLS, &internal_cells);
      two_points_scheme->setProperty(DiscreteOperatorProperty::P_FACES, &m_faces);
    }
    m_scheme->prepare(internal_faces, 
                      boundary_faces, 
                      m_c_internal_faces,
                      m_cf_internal_faces,
                      m_cell_coefficients.get(),
                      m_face_coefficients.get()) ;

    m_geometry->addItemGroupProperty(m_scheme->cells(), 
                                     m_scheme->getCellGeometricProperties(),IGeometryProperty::PItemGroupMap) ;
    m_geometry->addItemGroupProperty(m_scheme->faces(), 
                                     m_scheme->getFaceGeometricProperties(),IGeometryProperty::PItemGroupMap) ;
    m_geometry->addItemGroupProperty(m_scheme->cells(), 
                                     m_scheme->getCellGeometricProperties(),IGeometryProperty::PVariable) ;
    m_geometry->addItemGroupProperty(m_scheme->faces(), 
                                     m_scheme->getFaceGeometricProperties(),IGeometryProperty::PVariable) ;
    m_geometry->setPolicyTolerance(true);
    m_geometry->update();
    m_scheme->formDiscreteOperator(k);
  }

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
  SchemeType* m_scheme ;
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


template<typename SchemeType>
template<typename Collection1,typename Collection2>
void
FluxModelT<SchemeType>::
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

template<typename SchemeType>
template<typename Collection1,typename Collection2>
void
FluxModelT<SchemeType>::
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
template<typename SchemeType>
template<typename Collection1,typename Collection2, typename LambdaT>
void
FluxModelT<SchemeType>::
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

/*
#include "Numerics/DiscreteOperator/IDivKGradTwoPoints.h"
template<typename SchemeType>
template<typename TensorType>
void
FluxModelT<SchemeType>::
start(const TensorType& k)
{
  const CellGroup& internal_cells = m_numerical_domain->internalCells() ;
  const FaceGroup& internal_faces = m_numerical_domain->internalFaces() ;

  const FaceGroup& boundary_faces = m_numerical_domain->boundaryFaces() ;
  // Build the group of all the faces
  ItemGroupBuilder<Face> faces_builder(internal_faces.mesh(), IMPLICIT_NAME);
  ENUMERATE_FACE(iF, internal_faces) {
    const Face& F = *iF;
    faces_builder.add(F);
  }
  m_boundary_cells.init(boundary_faces) ;
  m_boundary_sgn.init(boundary_faces) ;
  ItemGroupSet is_internal_cell ;
  is_internal_cell.init(internal_cells) ;
  ENUMERATE_FACE(iF, boundary_faces) {
    const Face& F = *iF;
    faces_builder.add(F);
    if(F.isBoundary())
    {
      m_boundary_cells[F] = F.boundaryCell() ;
      m_boundary_sgn[F] = 1 ; ;
    }
    else
    {
      if(is_internal_cell.hasKey(F.backCell()))
      {
        m_boundary_cells[F] = F.backCell() ;
        m_boundary_sgn[F] = 1 ;
      }
      else
      {
        m_boundary_cells[F] = F.frontCell() ;
        m_boundary_sgn[F] = -1 ; 
      }
    }
  }
  m_faces = faces_builder.buildGroup();
  
  m_cell_coefficients.reset(new CoefficientArrayT<Cell>(m_faces, internal_cells));
  m_face_coefficients.reset(new CoefficientArrayT<Face>(m_faces, boundary_faces));
  
  m_c_internal_faces  = internal_faces.itemFamily()->createGroup(IMPLICIT_UNIQ_NAME);
  m_cf_internal_faces = internal_faces.itemFamily()->createGroup(IMPLICIT_UNIQ_NAME);

  IDivKGradTwoPoints * two_points_scheme = dynamic_cast<IDivKGradTwoPoints *>(m_scheme);
  if(two_points_scheme) {
    two_points_scheme->setProperty(DiscreteOperatorProperty::P_CELLS, &internal_cells);
    two_points_scheme->setProperty(DiscreteOperatorProperty::P_FACES, &m_faces);
  }
  m_scheme->prepare(internal_faces, 
                    boundary_faces, 
                    m_c_internal_faces,
                    m_cf_internal_faces,
                    m_cell_coefficients.get(),
                    m_face_coefficients.get()) ;

  m_geometry->addItemGroupProperty(m_scheme->cells(), 
                                   m_scheme->getCellGeometricProperties(),IGeometryProperty::PItemGroupMap) ;
  m_geometry->addItemGroupProperty(m_scheme->faces(), 
                                   m_scheme->getFaceGeometricProperties(),IGeometryProperty::PItemGroupMap) ;
  m_geometry->addItemGroupProperty(m_scheme->cells(), 
                                   m_scheme->getCellGeometricProperties(),IGeometryProperty::PVariable) ;
  m_geometry->addItemGroupProperty(m_scheme->faces(), 
                                   m_scheme->getFaceGeometricProperties(),IGeometryProperty::PVariable) ;
  m_geometry->setPolicyTolerance(true);
  m_geometry->update();
  m_scheme->formDiscreteOperator(k);
}
*/
#endif /*FLUXMODELT_H_*/
