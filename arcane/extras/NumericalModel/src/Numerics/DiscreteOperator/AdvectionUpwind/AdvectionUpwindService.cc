// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <arcane/ItemPrinter.h>
#include <vector>

#include <boost/static_assert.hpp>

#include "Numerics/DiscreteOperator/AdvectionUpwind/AdvectionUpwindService.h"

#include "Utils/ItemGroupBuilder.h"

#include "Appli/IAppServiceMng.h"
#include "Numerics/BCs/IBoundaryCondition.h"

using namespace Arcane;

void AdvectionUpwindService::init() {
  if( m_initialized ) return;

  // Retrieve and initialize application service manager
  IAppServiceMng* app_service_mng = IAppServiceMng::instance(subDomain()->serviceMng());
  
  // Retrieve shared geometry service  
  m_geometry_service = app_service_mng->find<IGeometryMng>(true);

  m_cells_group_name = IMPLICIT_UNIQ_NAME;
  m_faces_group_name = IMPLICIT_UNIQ_NAME;

  m_theta_upwind = options()->thetaUpwind();
  
  m_initialized = true;
}

////////////////////////////////////////////////////////////

void AdvectionUpwindService::prepare(const FaceGroup& internal_faces,
				     const FaceGroup& boundary_faces,
				     FaceGroup& c_internal_faces,
				     FaceGroup& cf_internal_faces,
				     CoefficientArrayT<Cell>* cell_coefficients,
				     CoefficientArrayT<Face>* face_coefficients) {

  if( !m_initialized )
    error() << " Numerical service not initialized";

  m_internal_faces = internal_faces;
  m_boundary_faces = boundary_faces;
  
  m_cell_coefficients = cell_coefficients;
  m_face_coefficients = face_coefficients;

  // Form face and cell groups
  ItemGroupBuilder<Face> faces_builder(m_internal_faces.mesh(), m_faces_group_name);
  faces_builder.add(m_internal_faces.enumerator());
  faces_builder.add(m_boundary_faces.enumerator());
  m_faces = faces_builder.buildGroup();

  ItemGroupBuilder<Face> c_internal_faces_builder(m_internal_faces.mesh(), 
                                                  c_internal_faces.name());
  c_internal_faces_builder.add(m_internal_faces.enumerator());
  m_c_internal_faces = c_internal_faces_builder.buildGroup();  

  ItemGroupBuilder<Face> cf_internal_faces_builder(m_internal_faces.mesh(),
                                                   cf_internal_faces.name());
  m_cf_internal_faces = cf_internal_faces_builder.buildGroup(); // Empty group

  ItemGroupBuilder<Cell> cells_builder(m_internal_faces.mesh(), m_cells_group_name);

  ItemGroupMapT<Face, Integer> cell_stencil_sizes(m_faces);
  ItemGroupMapT<Face, Integer> face_stencil_sizes(m_faces);

  ENUMERATE_FACE(iF, m_internal_faces) {
    const Face& F = *iF;

    cells_builder.add(F.cells());

    cell_stencil_sizes[F] = 2;
    face_stencil_sizes[F] = 0;
  }

  ENUMERATE_FACE(iF, m_boundary_faces) {
    const Face& F = *iF;

    cells_builder.add(F.cells());

    cell_stencil_sizes[F] = 1;
    face_stencil_sizes[F] = 1;
  }
  m_cells = cells_builder.buildGroup();  

  // Compute stencils
  m_cell_coefficients->init(cell_stencil_sizes);
  m_face_coefficients->init(face_stencil_sizes);

  ENUMERATE_FACE(iF, m_internal_faces) {
    const Face& F = *iF;

    ArrayView<Integer> stencil_F = m_cell_coefficients->stencilLocalId(F);
    stencil_F[0] = F.backCell().localId();
    stencil_F[1] = F.frontCell().localId();
  }
  
  ENUMERATE_FACE(iF, m_boundary_faces) {
    const Face& F = *iF;

    info() << ItemPrinter(F);
    
    ArrayView<Integer> c_stencil_F = m_cell_coefficients->stencilLocalId(F);
    ArrayView<Integer> f_stencil_F = m_face_coefficients->stencilLocalId(F);

    c_stencil_F[0] = F.boundaryCell().localId();
    f_stencil_F[0] = F.localId();
  }

  m_prepared = true;
}

////////////////////////////////////////////////////////////

void AdvectionUpwindService::finalize() {
  m_prepared = false;
}

////////////////////////////////////////////////////////////

void AdvectionUpwindService::formDiscreteOperator(const VelocityFluxType& q) {
  if(!m_prepared) error() << " Numerical service not prepared";

  ENUMERATE_FACE(iF, m_internal_faces) {
    const Face& F = *iF;

    // Retrieve neighbour cells
    // const Cell& T0 = F.backCell();
    // const Cell& T1 = F.frontCell();
    
    // Determine face orientation 
    // By convention n = n(back->front) 
    const Real alpha = 1.;
        
    // Get face velocity flux
    const Real& qF = q[F];
    
    // Build theta upwind scheme
    const Real& thetaF = m_theta_upwind ; 
    const Real a0 = (qF*alpha>0) ? thetaF : (1-thetaF);
    
    // Compute transmissivities    
    StencilFluxCoeffType tau_F = m_cell_coefficients->coefficients(F);
    tau_F[0] = a0*qF;
    tau_F[1] = (1-a0)*qF;
  }

  // Boundary faces
  ENUMERATE_FACE(iF, m_boundary_faces) {
    const Face& F = *iF;

    // Retrieve neighbour cell 
    const Cell& T0  = F.boundaryCell();
    
    // Determine output-face orientation 
    const Real alpha = (F.backCell() == T0) ? 1. : -1.;
           
    // Get face velocity flux (with output-face orientation)
    const Real& qF = q[F]*alpha;
        
    // Build theta upwind scheme
    const Real& thetaF  = m_theta_upwind; 
    Real a0 = (qF>0) ? thetaF : (1-thetaF);
    
    // Compute transmissivities
    StencilFluxCoeffType c_tau_F = m_cell_coefficients->coefficients(F);
    StencilFluxCoeffType f_tau_F = m_face_coefficients->coefficients(F);
    c_tau_F[0] = a0*qF;
    f_tau_F[0] = (1-a0)*qF;
  }
}

ARCANE_REGISTER_SERVICE_ADVECTIONUPWIND(AdvectionUpwind,
					AdvectionUpwindService);
