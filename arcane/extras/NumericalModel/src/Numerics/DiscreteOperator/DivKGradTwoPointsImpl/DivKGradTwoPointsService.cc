// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <vector>

#include "Numerics/DiscreteOperator/DivKGradTwoPointsImpl/DivKGradTwoPointsService.h"

#include "Appli/IAppServiceMng.h"

#include "Utils/ItemGroupBuilder.h"
// #include "Numerics/BCs/IBoundaryCondition.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/

void DivKGradTwoPointsService::setOption(const Integer& option)
{
  m_options |= option;
}

/*---------------------------------------------------------------------------*/

void DivKGradTwoPointsService::setProperty(const DiscreteOperatorProperty::eProperty& property, 
                                           const void * value) 
{
  switch(property) {
  case DiscreteOperatorProperty::P_CELLS: {
    const CellGroup * ptr = NULL;
    try {
      ptr = static_cast<const CellGroup *>(value);
    }
    catch(std::bad_cast) {
      error() << "Property P_CELLS requires a CellGroup to be passed";
    }

    m_cells = * ptr;

    m_properties |= DiscreteOperatorProperty::P_CELLS;

    break;
  }
  case DiscreteOperatorProperty::P_FACES: {
    const FaceGroup * ptr = NULL;
    try {
      ptr = static_cast<const FaceGroup *>(value);
    }
    catch(std::bad_cast) {
      error() << "Property P_FACES requires a FaceGroup to be passed";
    }
    m_faces = * ptr;

    m_properties |= DiscreteOperatorProperty::P_FACES;

    break;
  }
  default:
    error() << "Trying to set non existing property";
  }
}
/*---------------------------------------------------------------------------*/

void DivKGradTwoPointsService::init() 
{
  if( m_status & DiscreteOperatorProperty::S_INITIALIZED) 
    {
      return;
    }

  // Retrieve and initialize application service manager
  IAppServiceMng* app_service_mng = IAppServiceMng::instance(subDomain()->serviceMng());
  
  // Retrieve shared geometry service  
  m_geometry_service = app_service_mng->find<IGeometryMng>(true);

  m_cells_group_name = IMPLICIT_UNIQ_NAME;
  m_faces_group_name = IMPLICIT_UNIQ_NAME;

  m_status |= DiscreteOperatorProperty::S_INITIALIZED;
}

/*---------------------------------------------------------------------------*/

void DivKGradTwoPointsService::prepare(const FaceGroup& internal_faces,
                                       const FaceGroup& boundary_faces,
                                       FaceGroup& c_internal_faces,
                                       FaceGroup& cf_internal_faces,
                                       CoefficientArrayT<Cell>* cell_coefficients,
                                       CoefficientArrayT<Face>* face_coefficients) 
{
  ARCANE_ASSERT( (m_status & DiscreteOperatorProperty::S_INITIALIZED), 
                 ("Operator not initialized when calling prepare") );

  m_internal_faces = internal_faces;
  m_boundary_faces = boundary_faces;
  
  m_cell_coefficients = cell_coefficients;
  m_face_coefficients = face_coefficients;

  // Form face and cell groups
  if( !(m_properties & DiscreteOperatorProperty::P_FACES) )
    {
      warning() << "Building face group inside prepare(): this may be inefficient";

      ItemGroupBuilder<Face> faces_builder(m_internal_faces.mesh(), 
                                           m_faces_group_name);
      faces_builder.add(m_internal_faces.enumerator());
      faces_builder.add(m_boundary_faces.enumerator());
      m_faces = faces_builder.buildGroup();

      m_properties |= DiscreteOperatorProperty::P_FACES;
    }
  if( !(m_properties & DiscreteOperatorProperty::P_CELLS) )
    {
      warning() << "Building cell group inside prepare(): this may be inefficient";

      ItemGroupBuilder<Cell> cells_builder(m_internal_faces.mesh(), 
                                           m_cells_group_name);
      ENUMERATE_FACE(iF, m_internal_faces) {
        cells_builder.add(iF->cells());
      }

      //ENUMERATE_FACE(iF, m_boundary_faces) {
      //  cells_builder.add(iF->cells());
      //}
      m_cells = cells_builder.buildGroup();  

      m_properties |= DiscreteOperatorProperty::P_CELLS;
    }

  m_boundary_cells.init(boundary_faces) ;
  m_boundary_sgn.init(boundary_faces) ;
  ItemGroupSet is_internal_cell ;
  is_internal_cell.init(m_cells) ;
  ENUMERATE_FACE(iF, m_boundary_faces) {
    const Face& F = *iF;
    if(F.isBoundary())
      {
        m_boundary_cells[F] = F.boundaryCell() ;
        m_boundary_sgn[F] = 1 ; ;
      }
    else
      {
        if(is_internal_cell.hasKey(F.frontCell()))
          {
            m_boundary_cells[F] = F.frontCell() ;
            m_boundary_sgn[F] = -1 ;
          }
        else
          {
            m_boundary_cells[F] = F.backCell() ;
            m_boundary_sgn[F] = 1 ; ;
          }
      }
  }
  
  // c internal faces
  ItemGroupBuilder<Face> c_internal_faces_builder(m_internal_faces.mesh(), 
                                                  c_internal_faces.name());
  c_internal_faces_builder.add(m_internal_faces.enumerator());
  m_c_internal_faces = c_internal_faces_builder.buildGroup();  

  // cf internal faces
  ItemGroupBuilder<Face> cf_internal_faces_builder(m_internal_faces.mesh(),
                                                   cf_internal_faces.name());
  m_cf_internal_faces = cf_internal_faces_builder.buildGroup(); // Empty group

  ItemGroupBuilder<Cell> cells_builder(m_internal_faces.mesh(), m_cells_group_name);

  // Prepare stencils
  Array<std::pair<ItemGroup, Integer> > c_stencil_sizes(2);
  c_stencil_sizes[0].first  = m_internal_faces;
  c_stencil_sizes[0].second = 2;
  c_stencil_sizes[1].first  = m_boundary_faces;
  c_stencil_sizes[1].second = 1;

  Array<std::pair<ItemGroup, Integer> > f_stencil_sizes(2);
  f_stencil_sizes[0].first  = m_internal_faces;
  f_stencil_sizes[0].second = 0;
  f_stencil_sizes[1].first  = m_boundary_faces;
  f_stencil_sizes[1].second = 1;

  m_cell_coefficients->init(c_stencil_sizes);
  m_face_coefficients->init(f_stencil_sizes);

  // Compute stencils
  if( !(m_options & DiscreteOperatorProperty::O_DISABLE_STENCIL_COMPUTATION) )
    {
      ENUMERATE_FACE(iF, m_internal_faces) {
        const Face& F = *iF;

        ArrayView<Integer> stencil_F = m_cell_coefficients->stencilLocalId(F);
        stencil_F[0] = F.backCell().localId();
        stencil_F[1] = F.frontCell().localId();
      }
  
      ENUMERATE_FACE(iF, m_boundary_faces) {
        const Face& F = *iF;

        ArrayView<Integer> c_stencil_F = m_cell_coefficients->stencilLocalId(F);
        ArrayView<Integer> f_stencil_F = m_face_coefficients->stencilLocalId(F);

        //c_stencil_F[0] = F.boundaryCell().localId();
        c_stencil_F[0] = m_boundary_cells[F].localId();
        f_stencil_F[0] = F.localId();
      }
    }

  m_status |= DiscreteOperatorProperty::S_PREPARED;
}

/*---------------------------------------------------------------------------*/
void DivKGradTwoPointsService::finalize() 
{
  ARCANE_ASSERT( (m_status & DiscreteOperatorProperty::S_PREPARED),
                 ("Operator not prepared when calling finalize") );
  m_properties = DiscreteOperatorProperty::P_NONE;
  m_options    = DiscreteOperatorProperty::O_NONE;
  m_status     = DiscreteOperatorProperty::S_INITIALIZED;
}

/*---------------------------------------------------------------------------*/

void DivKGradTwoPointsService::formDiscreteOperator(const VariableCellReal& k) 
{
  _form_discrete_operator<Real>(k);
}

/*---------------------------------------------------------------------------*/

void DivKGradTwoPointsService::formDiscreteOperator(const VariableCellReal3& k) 
{
  _form_discrete_operator<Real3>(k);
}

/*---------------------------------------------------------------------------*/

void DivKGradTwoPointsService::formDiscreteOperator(const VariableCellReal3x3& k) 
{
  _form_discrete_operator<Real3x3>(k);
}

ARCANE_REGISTER_SERVICE_DIVKGRADTWOPOINTS(DivKGradTwoPoints,
                                          DivKGradTwoPointsService);
