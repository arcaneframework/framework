﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <sstream>

#include "Utils/Utils.h"
#include "Utils/ItemTools.h"
#include "Utils/ItemGroupBuilder.h"

#include "Appli/IAppServiceMng.h"

#include "Mesh/Geometry/IGeometryMng.h"

#include "Numerics/BCs/IBoundaryCondition.h"
#include "Numerics/BCs/IBoundaryConditionMng.h"

#include "ExpressionParser/IExpressionParser.h"
#include "ExpressionParser/IXYZTFunction.h"

#include "Numerics/LinearSolver/ILinearSolver.h"
#include "Numerics/LinearSolver/ILinearSystemVisitor.h"
#include "Numerics/LinearSolver/ILinearSystemBuilder.h"
#include "Numerics/LinearSolver/Impl/VariableUpdateImpl.h"

#include "Numerics/DiscreteOperator/IDiscreteOperator.h"
#include "Numerics/DiscreteOperator/IDivKGradDiscreteOperator.h"

#include "Numerics/LinearSolver/Impl/LinearSystemOneStepBuilder.h"

#include <cmath>

#include <iostream>
#include <fstream>

#include "Flow/FlowNumericalModel/BasicFlowModelService.h"

/* Author : haeberlf at Wed Aug 27 11:05:13 2008
 * Generated by createNew
 */

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
BasicFlowModelService::
init()
{
  //-----------------------------------------------------------------------------
  // Application Service Manager
  IServiceMng* service_mng = subDomain()->serviceMng();
  IAppServiceMng* app_service_mng = IAppServiceMng::instance(service_mng);

  //-----------------------------------------------------------------------------
  // Set Geometry Service
  info() << "Initialization Geometry Mng";
  m_geometry_service = app_service_mng->find<IGeometryMng>(true);

  //-----------------------------------------------------------------------------
  // Service Solveur Lineaire
  info() << "Get Option Linear Solver";
  m_linear_solver = options()->linearSolverFlow();

  //-----------------------------------------------------------------------------
  // Init Operator div(-D grad(u) )
  info() << "Get Flow Operator";
  m_op = options()->opflow();

  if( m_op == NULL)
    fatal() << "Discrete operator not defined";
  info() << "Flow Operator gotten";

  info() << "Initialize flow operator";
  m_op->init();
  info() << "Flow operator initialised";

  //-----------------------------------------------------------------------------
  // Init Flux Term Mng
  m_flux_term_service = options()->fluxTermService();
  m_flux_term_service->init();

  //-----------------------------------------------------------------------------
  // Init Boundary Conditions
  m_bc_flow_mng = options()->flowBoundaryConditionMng();
  m_bc_flow_mng->init();

  // -----------------------------------------------------------------------------
  // Build Face and Cell Groups
  ItemGroupBuilder<Cell> cell_builder(ownCells().mesh(), "CELLS");
  ENUMERATE_CELL(icell,ownCells())
  {
    const Cell& cell = *icell ;
    cell_builder.add(cell);
  }
  m_cells = cell_builder.buildGroup();
}

/*---------------------------------------------------------------------------*/

void
BasicFlowModelService::
prepare()
{
  // Prepare Flux Term Service
  m_flux_term_service->setOperator(m_op);
  m_flux_term_service->setBoundaryConditionMng(m_bc_flow_mng);
  m_flux_term_service->prepareOperator();
  m_flux_term_service->getOperatorGroups();

  // Prepare Linear solver, entries and equations
  prepareLinearSolver();
  prepareEquationsEntries();

  // form linear operator
  formOperator();
}

/*---------------------------------------------------------------------------*/
void BasicFlowModelService::
formOperator()
{
  m_op->formDiscreteOperator(m_flow_operator_tensor);
  info() << "Flow Operator Formed";
}

/*---------------------------------------------------------------------------*/
void
BasicFlowModelService::
prepareLinearSolver()
{
  //------------------------------------------------------------------------------
  // prepare the linear system builder and solver
  info() << "Linear system Set builder";
  m_linear_solver->setLinearSystemBuilder(&m_system_builder);

  info() << "Initialisation du systeme lineaire";
  m_linear_solver->init();
}

/*---------------------------------------------------------------------------*/

void
BasicFlowModelService::
prepareEquationsEntries()
{
  //------------------------------------------------------------------------------
  // Create boundary concentration variable for non-dirichlet boundary faces

  // groupe de faces de bord de type Dirichlet
  FaceGroup dirichlet_faces = m_bc_flow_mng->getFacesOfType(BoundaryConditionTypes::Dirichlet);

  // groupe de faces de bord de type non-Dirichlet
  FaceGroup non_dirichlet_faces = m_bc_flow_mng->getFacesNotOfType(BoundaryConditionTypes::Dirichlet);
  info() << outerFaces().size() - non_dirichlet_faces.size() << " Dirichlet faces found";
  info() << non_dirichlet_faces.size() << " Non-Dirichlet faces found";

  //------------------------------------------------------------------------------
  // Define Entries and Equations

  m_index_manager = m_system_builder.getIndexManager();
  m_index_manager->init();

  // creer les indices d'inconnus pour la concentration des mailles
  m_u_entry = m_index_manager->buildVariableEntry(m_cell_pressure.variable(),IIndexManager::Direct);

  // creer les indices des inconnus pour la concentrations au bord
  m_uboundary_entry = m_index_manager->buildVariableEntry(m_boundary_pressure.variable(),IIndexManager::Direct);
  info () << "Definition des indices de lignes";

  // creer les equations pour le transport entre les mailles
  m_flow_eq = m_index_manager->buildEquation("FlowEq", m_u_entry);

  // creer les equations pour les concentrations des faces de bord
  m_boundary_eq = m_index_manager->buildEquation("BoundaryEq", m_uboundary_entry);
  info () << "Fin de definition des equations";

  //------------------------------------------------------------------------------
  // Set Items for Entries and Equations

  // Cell balance equations
  m_index_manager->defineEquationIndex(m_flow_eq, m_cells.own());
  m_index_manager->defineEntryIndex(m_u_entry, m_cells);

  // Boundary equations
  m_index_manager->defineEquationIndex(m_boundary_eq, non_dirichlet_faces.own());
  m_index_manager->defineEntryIndex(m_uboundary_entry, non_dirichlet_faces);
}

/*---------------------------------------------------------------------------*/
void
BasicFlowModelService::
apply()
{
  // Solve Linear Equation
  computePressure();

  // Rebuild Explicit Fluxes
  computeVelocityFlux();

  // Interpolate Velocity Fluxes
  computeVelocity();
}

/*---------------------------------------------------------------------------*/
void
BasicFlowModelService::
clear()
{
  info() << "Clear Flow Model" ;
  m_flux_term_service->clearOperator();
}

/*---------------------------------------------------------------------------*/
void
BasicFlowModelService::checkFlow()
{
   // CHECK RESULTS
   ENUMERATE_CELL(icell,allCells())
    {
      const Cell& cell = *icell ;
      info() << "vx(" << int(cell.uniqueId())+1 << ") = " << m_cell_velocity[cell].x << "\n";
      info() << "vy(" << int(cell.uniqueId())+1 << ") = " << m_cell_velocity[cell].y << "\n";
      info() << "vz(" << int(cell.uniqueId())+1 << ") = " << m_cell_velocity[cell].z << "\n";
    }
    ENUMERATE_CELL(icell,allCells())
    {
      const Cell& cell = *icell ;
      info() << "pressure(" << int(cell.uniqueId())+1 << ") = " << m_cell_pressure[cell] << "\n";
    }

    ENUMERATE_FACE(iface,allFaces())
    {
      const Face& face = *iface ;
      info() << "normalflux(" << int(face.uniqueId())+1 << ") = " << m_face_normal_flux_velocity[face];
    }
}

/*---------------------------------------------------------------------------*/

void BasicFlowModelService::computePressure()
{
  //------------------------------------------------------------------------------
  // Build The Linear System

  // initialiser le systeme lineaire
  info() << "Assembling linear system";

  m_system_builder.start();
  m_linear_solver->start();


  // Add Flux Term Contribution for flow Equation
  m_flux_term_service->addCInternalFacesContribution(&m_system_builder,
                                                      m_index_manager,
                                                      m_flow_eq,
                                                      m_u_entry);

  m_flux_term_service->addCFInternalFacesContribution(&m_system_builder,
                                                      m_index_manager,
                                                      m_flow_eq,
                                                      m_u_entry,
                                                      m_uboundary_entry);

  m_flux_term_service->addBoundaryFacesContribution(&m_system_builder,
                                                      m_index_manager,
                                                      m_flow_eq,
                                                      m_u_entry,
                                                      m_uboundary_entry);

  // Add Flux Term Contribution for boundary Equation
  m_flux_term_service->addBoundaryNonDirichletFacesContribution(&m_system_builder,
                                                      m_index_manager,
                                                      m_boundary_eq,
                                                      m_u_entry,
                                                      m_uboundary_entry);

  //  m_system_builder.dumpToMatlab("matlab_flow_data.m");


  //------------------------------------------------------------------------------
  // Solve Linear System and get solution

  info() << "Solve The Linear System ";
  m_linear_solver->buildLinearSystem() ;
  bool success = m_linear_solver->solve();
  if (not success)
    fatal() << "Solver failed to converge";
  else
    info() << "Convergence reached";
  m_linear_solver->getSolution();

  // Delete the  Linear System
  m_linear_solver->end();
  m_system_builder.end();
}

/*---------------------------------------------------------------------------*/
void BasicFlowModelService::computeVelocityFlux()
{
  // Compute Flux Term values
  m_flux_term_service->computeCInternalFacesFluxValues(
                         &m_face_normal_flux_velocity,
                         m_cell_pressure);

  m_flux_term_service->computeCFInternalFacesFluxValues(
                         &m_face_normal_flux_velocity,
                         m_cell_pressure,
                         m_boundary_pressure);

  m_flux_term_service->computeBoundaryFacesFluxValues(
                         &m_face_normal_flux_velocity,
                         m_cell_pressure,
                         m_boundary_pressure);

}

/*---------------------------------------------------------------------------*/
void BasicFlowModelService::computeVelocity()
{
    IInterpolator* interpolator = options()->interpolator();
    interpolator->init();

    m_geometry_service->addItemGroupProperty(ownCells(), interpolator->getCellGeometricProperties());
    m_geometry_service->addItemGroupProperty(allFaces(), interpolator->getFaceGeometricProperties());
    m_geometry_service->update();

    // Interpolate fields
    info() << "Interpolating velocity Field";
    interpolator->setSourceField(&m_face_normal_flux_velocity);
    interpolator->setTargetField(&m_cell_velocity);

    interpolator->interpolate();
    m_cell_velocity.synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_BASICFLOWMODEL(BasicFlowModel,BasicFlowModelService);
