// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ADVECTIONUPWINDSERVICE_H
#define ADVECTIONUPWINDSERVICE_H

#include <map>

#include <boost/shared_ptr.hpp>

#include "Mesh/Geometry/IGeometryMng.h"
#include "Numerics/DiscreteOperator/CoefficientArray.h"
#include "Numerics/DiscreteOperator/DiscreteOperatorProperty.h"

#include "Numerics/DiscreteOperator/IAdvectionOperator.h"

#include "AdvectionUpwind_axl.h"

using namespace Arcane;

/*!
  \class Theta Upwind discretization of the \f$\nabla\cdot(v \cdot)\f$
  operator
  \author Anthony Michel
  \date 2008-02-19
*/
class AdvectionUpwindService : public ArcaneAdvectionUpwindObject {
 public:
  //! Constructor
  AdvectionUpwindService(const ServiceBuildInfo& sbi) :
    ArcaneAdvectionUpwindObject(sbi),
    m_initialized(false),
    m_prepared(false) {}

  //! Destructor
  virtual ~AdvectionUpwindService() {
    // Delete m_face group   
  }

  //! Return version number
  virtual VersionInfo versionInfo() const { return VersionInfo(1, 0, 0); }

 public:
  //! Initialize the service
  void init();

  //! Preliminary computations (including connectivity)
  void prepare(const FaceGroup& internal_faces,
	       const FaceGroup& boundary_faces,
	       FaceGroup& c_internal_faces,
	       FaceGroup& cf_internal_faces,
	       CoefficientArrayT<Cell>* cell_coefficients,
	       CoefficientArrayT<Face>* face_coefficients);
  //! Reset the operator
  void finalize();

  //! Form the discrete operator associated with the face velocity flux \f$q\f$ 
  //! parametrized by the face upwinding parameter \f$\theta\f$
  void formDiscreteOperator(const VelocityFluxType& q);

  //! Required cell geometric properties
  inline Integer getCellGeometricProperties() { 
    return IGeometryProperty::PCenter; 
  }
  //! Required face geometric properties
  inline Integer getFaceGeometricProperties() { 
    return (IGeometryProperty::PCenter | IGeometryProperty::PNormal); 
  }

  public:
  //! Return faces
  const FaceGroup& faces() const { return m_faces; }
  //! Return cells 
  const CellGroup& cells() const { return m_cells; }
  //! Return boundary faces
  const FaceGroup& boundaryFaces() const { return m_boundary_faces; }
  //! Return internal faces
  const FaceGroup& internalFaces() const { return m_internal_faces; }
  //! Return internal faces whose stencil contains cell unknowns only
  const FaceGroup& cInternalFaces() const { return m_c_internal_faces; }
  //! Return internal faces whose stencil contains both cells and face unknowns
  const FaceGroup& cfInternalFaces() const { return m_cf_internal_faces; }

  private:
 
  //! Geometry service
  IGeometryMng* m_geometry_service;

  //! Initialized flag
  bool m_initialized;
  
  //! Prepared flag
  bool m_prepared;

  //! Faces group
  FaceGroup m_faces;
  //! Cells group
  CellGroup m_cells;
  //! Faces whose flux stencil contains cell unknowns only
  FaceGroup m_internal_faces;
  //! Boundary faces group
  FaceGroup m_boundary_faces;
  //! Internal faces whose stencil contains cell unknowns only
  FaceGroup m_c_internal_faces;
  //! Internal faces whose stencil contains both cells and face unknowns
  FaceGroup m_cf_internal_faces;

  //! Cell group name
  String m_cells_group_name;
  //! Face group name
  String m_faces_group_name;

  //! Cell coefficients
  CoefficientArrayT<Cell>* m_cell_coefficients;
  //! Face coefficients
  CoefficientArrayT<Face>* m_face_coefficients;
  
  //! Upwinding parameter
  Real m_theta_upwind;
};

#endif
