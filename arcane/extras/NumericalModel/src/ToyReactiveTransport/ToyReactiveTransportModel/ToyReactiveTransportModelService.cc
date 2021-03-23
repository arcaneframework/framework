// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "Utils/Utils.h"

#include "Utils/ItemGroupMap.h"
#include "Utils/ItemGroupBuilder.h"

#include "NumericalModel/Utils/ICollector.h"
#include "NumericalModel/Utils/IOp.h"
#include "NumericalModel/Models/INumericalModel.h"
#include "NumericalModel/Operators/INumericalModelVisitor.h"
#include "NumericalModel/SubDomainModel/INumericalDomain.h"
#include "NumericalModel/SubDomainModel/NumericalDomain/NumericalDomainImpl.h"
#include "NumericalModel/SubDomainModel/SubDomainModelProperty.h"
#include "NumericalModel/Utils/BaseCollector.h"
#include "NumericalModel/Utils/OpT.h"
#include "NumericalModel/Utils/BaseModelOpT.h"
#include "NumericalModel/Utils/TimeStepModelOpT.h"
#include "NumericalModel/SubDomainModel/CollectorT.h"
#include "NumericalModel/SubDomainModel/SDMBoundaryCondition.h"
#include "NumericalModel/SubDomainModel/SDMBoundaryConditionMng.h"
#include "NumericalModel/Models/ISubDomainModel.h"

#include "Numerics/DiscreteOperator/IDivKGradDiscreteOperator.h"
#include "Numerics/DiscreteOperator/IAdvectionOperator.h"

#include "Numerics/LinearSolver/ILinearSolver.h"

#include "Mesh/Interpolator/IInterpolator.h"

#include "Appli/IAppServiceMng.h"

#include "Mesh/Geometry/IGeometryMng.h"

#include "Numerics/LinearSolver/IIndexManager.h"
#include "Numerics/LinearSolver/ILinearSystemVisitor.h"
#include "Numerics/LinearSolver/Impl/LinearSystemOneStepBuilder.h"
#include "Numerics/LinearSolver/Impl/LinearSystemTwoStepBuilder.h"

#include "Numerics/Expressions/IExpressionMng.h"
#include "Numerics/Expressions/IFunctionR4vR1.h"
#include "Numerics/Expressions/IFunctionR3vR1.h"
#include "Numerics/Expressions/FunctionParser/FunctionParser.h"
#include "Numerics/Expressions/ExpressionBuilder/ExpressionBuilderR4vR1Core.h"
#include "Numerics/Expressions/ExpressionBuilder/ExpressionBuilderR3vR1Core.h"
#include "Numerics/Expressions/ExpressionBuilder/ExpressionBuilderR1vR1Core.h"

#include "TimeUtils/ITimeMng.h"
#include "TimeUtils/ITimeStepMng.h"

#include "Utils/ItemGroupMap.h"
#include "Utils/MeshVarExpr.h"
#include "NumericalModel/Algorithms/ITimeIntegrator.h"
#include "NumericalModel/Algorithms/FiniteVolumeAlgo/LinearSystemAssembleOp.h"
#include "NumericalModel/FluxModel/FluxModel.h"
#include "NumericalModel/FluxModel/FluxModelT.h"
#include "NumericalModel/Utils/TimeStepModelOpT.h"
#include "NumericalModel/Utils/TimeIntegratorOpT.h"

#include "Flow/FlowNumericalModel/IFlowNumericalModel.h"

#include "Mesh/GroupCreator/IGroupCreator.h"

#include "Mesh/Utils/MeshUtils.h"

#include <arcane/IVariable.h>
#include <arcane/IVariableAccessor.h>

#include "Numerics/DiscreteOperator/CoefficientArray.h"

#include <boost/shared_ptr.hpp>
#include "Utils/ItemGroupBuilder.h"
#include "ToyReactiveTransport/ToyReactiveTransportModel/ToyReactiveTransportModelService.h"

#define NEUMANN_BC SubDomainModelProperty::Neumann
#define DIRICHLET_BC SubDomainModelProperty::Dirichlet
#define OVERLAPDIRICHLET_BC SubDomainModelProperty::OverlapDirichlet

using namespace Arcane;
using namespace MeshVariableOperator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

INumericalDomain*
ToyReactiveTransportModelService::getINumericalDomain()
  {
    return getNumericalDomain();
  }

ISubDomainModel::NumericalDomain*
ToyReactiveTransportModelService::getNumericalDomain()
  {
    if (m_numerical_domain == NULL)
      m_numerical_domain = new ISubDomainModel::NumericalDomain(traceMng(),
          this);
    return m_numerical_domain;
  }

ToyReactiveTransportModelService::FaceBoundaryConditionMng*
ToyReactiveTransportModelService::getFaceBoundaryConditionMng()
  {
    if (m_bc_mng == NULL)
      {
        m_bc_mng = new FaceBoundaryConditionMng(getNumericalDomain());
      }
    return m_bc_mng;
  }
void ToyReactiveTransportModelService::init()
  {
    if (m_initialized)
      return;
    m_output_level = options()->outputLevel();
    // Retrieve and initialize application service manager
    IServiceMng* service_mng = subDomain()->serviceMng();
    IAppServiceMng* app_service_mng = IAppServiceMng::instance(service_mng);

    // Retrieve shared geometry service
    m_geometry = app_service_mng->find<IGeometryMng> (true);
    m_geometry->setPolicyTolerance(true);

    m_diff_flux_scheme = options()->diffSchemeFluxDomain();
    m_diff_flux_scheme->init();

    m_adv_flux_scheme = options()->advSchemeFluxDomain();
    m_adv_flux_scheme->init();

    m_time_step_mng = options()->timeStepMng();
    m_time_step_mng->init();

    m_interpolator = options()->interpolator();
    m_interpolator->init();
    m_interpolator->prepare();

    // Création des groupes (pouvant être utilisés sur les bords)
    for (Integer i = 0; i < options()->groupCreator.size(); ++i)
      {
        options()->groupCreator[i]->init();
      }

    m_expression_mng = app_service_mng->find<IExpressionMng> (true);
    if (m_expression_mng == NULL)
      fatal() << "Expression Mng service not found";
    //m_expression_mng->init();
    m_local_expression_mng = NULL;

    info() << "get Function Parser";
    //m_function_parser = new FunctionParser();
    m_rhs_function_parser.init(m_expression_mng, m_local_expression_mng,
        traceMng());
    m_k_function_parser.init(m_expression_mng, m_local_expression_mng,
        traceMng());
    m_psi_function_parser.init(m_expression_mng, m_local_expression_mng,
        traceMng());

    m_vars.setUConcentration(&m_u_concentration);
    m_vars.setVConcentration(&m_v_concentration);

    m_geometry->setPolicyTolerance(true);
    m_geometry->update(); /// misplaced update : doit etre délégué à celui qui voit l'évolution de la géométrie.
    // Création des groupes (pouvant être utilisés sur les bords)
    for (Integer i = 0; i < options()->groupCreator.size(); ++i)
      {
        options()->groupCreator[i]->apply();
      }

    m_error = 0;
    m_initialized = true;
  }

void ToyReactiveTransportModelService::start()
  {
    getNumericalDomain();

    //Create Flux model and init Discrete Operators
    if (m_diff_flux_model == NULL)
      {
        m_diff_flux_model = new DiffFluxModelType(m_numerical_domain,
            m_diff_flux_scheme, m_geometry);
        m_diff_flux_model->init();
      }
    if (m_adv_flux_model == NULL)
      {
        m_adv_flux_model = new AdvFluxModelType(m_numerical_domain,
            m_adv_flux_scheme, m_geometry);
        m_adv_flux_model->init();
      }

    const CellGroup& internal_cells = m_numerical_domain->internalCells();
    ENUMERATE_CELL(icell, internal_cells)
      {
        m_cell_perm_k[icell].x.x = m_cell_permx[icell];
        m_flow_operator_tensor[icell].x.x = m_permeability[icell];
        m_cell_perm_k[icell].y.y = m_cell_permy[icell];
        m_flow_operator_tensor[icell].y.y = m_permeability[icell];
        m_cell_perm_k[icell].z.z = m_cell_permz[icell];
        m_flow_operator_tensor[icell].z.z = m_permeability[icell];
      }
    CellGroup cellgroup = ownCells().itemFamily()->findGroup("Barriers", false);
    if (!cellgroup.null())
      {
        ENUMERATE_CELL(icell, cellgroup)
          {
            m_flow_operator_tensor[icell].x.x = 1.;
            m_flow_operator_tensor[icell].y.y = 1.;
            m_flow_operator_tensor[icell].z.z = 1.;
          }
      }
    m_diff_flux_model->start(m_cell_perm_k);

    // Flow Model
    info() << "compute Flow";
    if (options()->flowModel.size() > 0)
      {
        IFlowNumericalModel* flow_model = options()->flowModel[0];
        flow_model->init();
        flow_model->prepare();
        flow_model->apply();
        info() << "compute Flow Model Done.";
        m_cell_velocity.synchronize();
        m_face_normal_flux_velocity.synchronize();

      }
    else
      {
        info() << "No Flow Model !!";
      }
    m_adv_flux_model->start(m_face_normal_flux_velocity);

    //init Boundaries conditions
    if (m_bc_mng)
      {
        m_bc_values.init(m_numerical_domain->boundaryFaces());
        m_bc_semi_values.init(m_numerical_domain->boundaryFaces());
        m_bc_mng->initBC();
      }
    m_geometry->addItemGroupProperty(internal_cells,
        IGeometryProperty::PVolume, IGeometryProperty::PVariable);
    m_geometry->addItemGroupProperty(internal_cells,
        IGeometryProperty::PCenter, IGeometryProperty::PVariable);
    m_geometry->update(); /// misplaced update : doit etre délégué à celui qui voit l'évolution de la géométrie.

    m_error = 0;
  }

typedef TimeStepModelOpT<ToyReactiveTransportModelService> TimeOp;
typedef TimeIntegratorOpT<ToyReactiveTransportModelService> TimeIntegratorOp;
typedef BaseModelOpT<ToyReactiveTransportModelService> BaseOp;
void ToyReactiveTransportModelService::prepare(ICollector* collector,
    Integer sequence)
  {
    CollectorListIter iter = m_collectors.find(sequence);
    if (iter == m_collectors.end())
      {
        m_collectors[sequence] = collector;
        ISubDomainModel::Collector* col =
            dynamic_cast<ISubDomainModel::Collector*> (collector);
        if (col)
          {
            Visitor* op = new TimeOp(col->getTimeMng(), collector);
            Sequence base_seq(BaseSeq, op);
            if (col->useTimeIntegrator())
              {
                //Create an integrator op and record the Time op operator on slot -1-sequence
                ITimeIntegrator* time_integrator = col->getTimeIntegrator();
                Visitor* integrator_op = new TimeIntegratorOp(-1 - sequence,
                    time_integrator);
                Sequence seq(AlgoSeq, integrator_op);
                m_sequences[sequence] = seq;
                m_sequences[-1 - sequence] = base_seq;
              }
            else
              m_sequences[sequence] = base_seq;
          }
        else
          {
            Visitor* op = new BaseOp(collector);
            m_sequences[sequence] = Sequence(BaseSeq, op);
          }
      }
    else
      {
        fatal() << "Multi collector preparation not yet implemented";
        //(*iter).second->update() ;
        //m_collector_ops[sequence]->update() ;
      }
  }
/*---------------------------------------------------------------------------*/
void ToyReactiveTransportModelService::startTimeStep()
  {
    m_current_time = m_time_mng->getCurrentTime();
    m_current_time_step = m_time_mng->getCurrentTimeStep();
    if (m_output_level > 0)
      {
        info();
        info() << "----------------------------------------------";
        info() << "| Current Time      :" << m_current_time;
        info() << "| Current Time Step :" << m_current_time_step;
        info() << "----------------------------------------------";
      }
  }

void ToyReactiveTransportModelService::start(Integer sequence)
  {
    m_error += acceptForStart(m_sequences[sequence].m_op);
  }

Integer ToyReactiveTransportModelService::compute(Integer sequence)
  {
    debug(Trace::Medium) << "compute SEQUENCE" << sequence;
    Sequence& seq = m_sequences[sequence];
    switch (seq.m_type)
      {
    case BaseSeq:
      debug(Trace::Medium) << "compute Base SEQUENCE";
      m_error += acceptForStart(seq.m_op);
      m_error += accept(seq.m_op);
      m_error += acceptForFinalize(seq.m_op);
      break;
    case AlgoSeq:
      debug(Trace::Medium) << "compute ALGO SEQUENCE";
      m_error += accept(seq.m_op);
      break;
    default:
      {
        ICollector* col = m_collectors[sequence];
        col->start();
        col->computeAll();
        col->finalize();
      }
      }
    return m_error;
  }
Integer ToyReactiveTransportModelService::baseCompute(Integer sequence)
  {
    notifyNewSequence(sequence);
    m_error += accept(m_sequences[sequence].m_op);
    return m_error;
  }

void ToyReactiveTransportModelService::finalize(Integer sequence)
  {
    acceptForFinalize(m_sequences[sequence].m_op);
  }

template<class Field>
Integer ToyReactiveTransportModelService::computeT(Field& u_concentration,
    Field& v_concentration)
  {
    const CellGroup internal_cells = m_numerical_domain->internalCells();

    // Retrieve geometrical properties
    const IGeometryMng::RealVariable & measures =
        m_geometry->getRealVariableProperty(internal_cells,
            IGeometryProperty::PVolume);
    ArcGeoSim::Mesh::assign(m_phirhocp, measures * m_rho * m_compressibility,
        internal_cells);

    const IGeometryMng::Real3Variable & cell_centers =
        m_geometry->getReal3VariableProperty(allCells(),
            IGeometryProperty::PCenter);

    //============== start rhs construction ================================

    String rhs_expression = options()->rhs();
    rhs_expression = "(x,y,z,t)-> " + rhs_expression;
    info() << "  expression=" << rhs_expression;

    info() << "  Parse expression";
    m_rhs_function_parser.parseString(rhs_expression); //"(x,y,z,t)-> x+y*-sin(z)+t" );

    info() << "  Build Function filter";
    ExpressionBuilderR4vR1Core rhs_func = ExpressionBuilderR4vR1Core(
        &m_rhs_function_parser);

    // recopy
    // to be removed when IFunctionR3vR1 interCell implements eval(Real3,Real) method
    Array<Real> x(internal_cells.size());
    Array<Real> y(internal_cells.size());
    Array<Real> z(internal_cells.size());
    Array<Real> t(internal_cells.size());
    Array<Real> r(internal_cells.size());
    Integer iData = 0;
    ENUMERATE_CELL(iCell, internal_cells)
      {
        const Cell& cell = *iCell;
        const Real3& center = cell_centers[cell];

        x[iData] = center.x;
        y[iData] = center.y;
        z[iData] = center.z;
        t[iData] = m_current_time;
        iData++;
      }

    // Evaluate rhs
    rhs_func.eval(x, y, z, t, r);

    // and recopy... very very no good
    iData = 0;

    ENUMERATE_CELL (iCell, internal_cells)
      {
        const Cell& cell = *iCell;

        m_cell_rhs_value[cell] = r[iData];
        iData++;
      }

    //================= end rhs construction ================================


    //============== start reactive term construction ================================

    String k_expression = options()->k();
    k_expression = "(x,y,z,t)-> " + k_expression;
    info() << "  expression=" << k_expression;

    info() << "  Parse expression";
    m_k_function_parser.parseString(k_expression); //"(x,y,z,t)-> x+y*-sin(z)+t" );

    info() << "  Build Function filter";
    ExpressionBuilderR4vR1Core k_func = ExpressionBuilderR4vR1Core(
        &m_k_function_parser);

    String psi_expression = options()->psi();
    psi_expression = "(u)-> " + psi_expression;
    info() << "  expression=" << psi_expression;

    info() << "  Parse expression";
    m_psi_function_parser.parseString(psi_expression); //"(u)-> u*u + u - 5" );

    info() << "  Build Function filter";
    ExpressionBuilderR1vR1Core psi_func = ExpressionBuilderR1vR1Core(
        &m_psi_function_parser);

    // recopy
    // to be removed when IFunctionR3vR1 interCell implements eval(Real3,Real) method
    Array<Real> k(internal_cells.size());
    Array<Real> psi(internal_cells.size());
    Array<Real> u(internal_cells.size());
    iData = 0;
    ENUMERATE_CELL(iCell, internal_cells)
      {
        const Cell& cell = *iCell;
        const Real3& center = cell_centers[cell];

        x[iData] = center.x;
        y[iData] = center.y;
        z[iData] = center.z;
        u[iData] = m_u_concentration[cell];
        t[iData] = m_current_time;
        iData++;
      }

    // Evaluate rhs
    k_func.eval(x, y, z, t, k);
    psi_func.eval(u, psi);

    // and recopy... very very no good
    iData = 0;

    ENUMERATE_CELL (iCell, internal_cells)
      {
        const Cell& cell = *iCell;

        m_cell_k_value[cell] = k[iData];
        m_cell_r_value[cell] = m_cell_k_value[cell] * (m_v_concentration[cell]
            - psi[iData]);
        //update v
        iData++;
      }

    // Solve decoupled v
    ENUMERATE_CELL (iCell, internal_cells)
      {
        const Cell& cell = *iCell;
        //update v
        m_v_concentration[cell] -= m_current_time_step * m_cell_r_value[cell];
      }

    //================= end reactive term construction ================================


    ILinearSolver *solver = options()->linearsolver();
    LinearSystemOneStepBuilder builder;
    LinearSystemOneStepBuilder *system = &builder;

    solver->setLinearSystemBuilder(&builder);
    solver->init();

    // numerotation des inconnes de mailles
    IIndexManager * manager = system->getIndexManager();

    IIndexManager::Entry uEntry = manager->buildVariableEntry(
        u_concentration.variable(), IIndexManager::Direct);
    IIndexManager::Entry bEntry = manager->buildVariableEntry(
        m_flux_overlap.variable(), IIndexManager::Direct);
    IIndexManager::Equation uEquation = manager->buildEquation(
        "UConcentration", uEntry);

    // Cell balance equations
    manager->defineEquationIndex(uEquation, internal_cells.own());
    manager->defineEntryIndex(uEntry, internal_cells);

    // Boundary equations
    //    manager->defineEquationIndex(bnd_eqn, m_numerical_domain->boundaryFaces());
    //    manager->defineEntryIndex(bEntry, m_numerical_domain->boundaryFaces());

    system->start();

    ScalarMeshVar one(1.);
    Real invdt = 1. / m_current_time_step;
    //MassTerm<LinearSystemOneStepBuilder>,MassExprType,RHSExprType>
    MassTerm<LinearSystemOneStepBuilder>::assemble(builder, uEntry, uEquation,
        internal_cells, (invdt * m_phirhocp), (invdt * m_phirhocp)
            * u_concentration + m_cell_rhs_value + m_cell_r_value);

    FluxTerm<LinearSystemOneStepBuilder, DiffFluxModelType>::assemble(builder,
        uEntry, uEquation, m_diff_flux_model, m_rho);
    FluxTerm<LinearSystemOneStepBuilder, AdvFluxModelType>::assemble(builder,
        uEntry, uEquation, m_adv_flux_model, m_rho);

    for (FaceBoundaryConditionMng::BoundaryConditionIter iter =
        m_bc_mng->getBCIter(); iter.notEnd(); ++iter)
      {
        FaceBoundaryConditionMng::BoundaryCondition* bc = (*iter);
        if (bc->isActive())
          {
            const FaceGroup& boundary = (*iter)->getBoundary();
            debug(Trace::Medium) << "Boundary Condition " << bc->getType()
                << " " << boundary.name();
            switch (bc->getType())
              {
            case NEUMANN_BC:
              {
                m_flux_ratio.synchronize();
                ENUMERATE_FACE(iF, boundary)
                  {
                    const Face& F = *iF;
                    //                    if (m_flux_ratio[F]!=NULL)
                    //                      m_bc_semi_values[F] = (1. - m_flux_ratio[F])
                    //                          * m_bc_values[F];
                    //                    else
                    m_bc_semi_values[F] = 0.5 * m_bc_values[F];
                  }
                TwoPtsBCFluxTerm<LinearSystemOneStepBuilder, DiffFluxModelType,
                    SubDomainModelProperty, NEUMANN_BC>::assemble(builder,
                    uEntry, uEquation, bEntry, m_diff_flux_model, boundary,
                    m_diff_flux_model->getBoundaryCells(),
                    m_diff_flux_model->getBoundarySgn(), m_rho,
                    m_bc_semi_values);
                //                ENUMERATE_FACE(iF, boundary)
                //                  {
                //                    const Face& F = *iF;
                //                    if (m_flux_ratio[F]!=NULL)
                //                      m_bc_semi_values[F] = m_flux_ratio[F] * m_bc_values[F];
                //                    else
                //                      m_bc_semi_values[F] = 0.5 * m_bc_values[F];
                //                  }
                TwoPtsBCFluxTerm<LinearSystemOneStepBuilder, AdvFluxModelType,
                    SubDomainModelProperty, NEUMANN_BC>::assemble(builder,
                    uEntry, uEquation, bEntry, m_adv_flux_model, boundary,
                    m_adv_flux_model->getBoundaryCells(),
                    m_adv_flux_model->getBoundarySgn(), m_rho, m_bc_semi_values);

                break;
              }
            case DIRICHLET_BC:
              {
                TwoPtsBCFluxTerm<LinearSystemOneStepBuilder, DiffFluxModelType,
                    SubDomainModelProperty, DIRICHLET_BC>::assemble(builder,
                    uEntry, uEquation, bEntry, m_diff_flux_model, boundary,
                    m_diff_flux_model->getBoundaryCells(),
                    m_diff_flux_model->getBoundarySgn(), m_rho, m_bc_values);
                TwoPtsBCFluxTerm<LinearSystemOneStepBuilder, AdvFluxModelType,
                    SubDomainModelProperty, DIRICHLET_BC>::assemble(builder,
                    uEntry, uEquation, bEntry, m_adv_flux_model, boundary,
                    m_adv_flux_model->getBoundaryCells(),
                    m_adv_flux_model->getBoundarySgn(), m_rho, m_bc_values);
                break;
              }
            case OVERLAPDIRICHLET_BC:
              {
                TwoPtsBCFluxTerm<LinearSystemOneStepBuilder, DiffFluxModelType,
                    SubDomainModelProperty, DIRICHLET_BC>::assemble(builder,
                    uEntry, uEquation, bEntry, bc->getDiffFluxModel(),
                    boundary, m_diff_flux_model->getBoundaryCells(),
                    m_diff_flux_model->getBoundarySgn(), m_rho, m_bc_values);
                TwoPtsBCFluxTerm<LinearSystemOneStepBuilder, AdvFluxModelType,
                    SubDomainModelProperty, DIRICHLET_BC>::assemble(builder,
                    uEntry, uEquation, bEntry, bc->getAdvFluxModel(), boundary,
                    m_adv_flux_model->getBoundaryCells(),
                    m_adv_flux_model->getBoundarySgn(), m_rho, m_bc_values);

                break;
              }
            default:
              break;
              }
          }
      }
    // Resolution du system lineaire
    solver->buildLinearSystem();
    const bool success = solver->solve();
    if (not success)
      fatal() << "Solver Convergence failed";
    solver->getSolution(); // commitSolution
    solver->end(); // Appel builder.end()
    /* DEBUG
    if (m_output_level > 2)
      {
        ENUMERATE_CELL (icell,internal_cells)
          {
            info() << "Sol[" << icell->uniqueId() << "]="
                << m_u_concentration[*icell];
            //debug(Trace::Medium)<<"Sol["<<icell->uniqueId()<<"]="<<pressure[*icell] ;
          }
      }
    */
    return 0;
  }
Integer ToyReactiveTransportModelService::baseCompute()
  {
    return computeT<VariableCellReal> (*m_vars.getUConcentration(),
        *m_vars.getVConcentration());
  }

void ToyReactiveTransportModelService::setVars(ICollector* collector)
  {
    CollectorT<ISubDomainModel>* col =
        dynamic_cast<CollectorT<ISubDomainModel>*> (collector);
    if (col)
      {
        VariableCellReal* u_concentration = col->getRealVar(
            SubDomainModelProperty::UConcentration);
        if (u_concentration)
          m_vars.setUConcentration(u_concentration);
        VariableCellReal* v_concentration = col->getRealVar(
            SubDomainModelProperty::VConcentration);
        if (v_concentration)
          m_vars.setVConcentration(v_concentration);
      }
  }

Integer ToyReactiveTransportModelService::accept(
    INumericalModelVisitor* visitor)
  {
    Visitor* ptr = dynamic_cast<Visitor*> (visitor);
    if (ptr)
      m_error += ptr->visit(this);
    else
      m_error += visitor->visit((IDomainModel*) this);
    return m_error;
  }
Integer ToyReactiveTransportModelService::acceptForStart(
    INumericalModelVisitor* visitor)
  {
    Visitor* ptr = dynamic_cast<Visitor*> (visitor);
    if (ptr)
      m_error += ptr->visitForStart(this);
    else
      m_error += visitor->visit((IDomainModel*) this);
    return m_error;
  }
Integer ToyReactiveTransportModelService::acceptForFinalize(
    INumericalModelVisitor* visitor)
  {
    Visitor* ptr = dynamic_cast<Visitor*> (visitor);
    if (ptr)
      m_error += ptr->visitForFinalize(this);
    else
      m_error += visitor->visit((IDomainModel*) this);
    return m_error;
  }

void ToyReactiveTransportModelService::initBoundaryCondition(
    ToyReactiveTransportModelService::FaceBoundaryCondition* bc)
  {
    debug(Trace::Medium) << "Init BC" << bc->getId();
    bc->initValues(m_bc_values);
    bc->initValues(m_bc_semi_values);
  }

void ToyReactiveTransportModelService::initBoundaryCondition()
  {
    if (m_bc_mng)
      {
        for (FaceBoundaryConditionIter iter = m_bc_mng->getBCIter(); iter.notEnd(); ++iter)
          {
            FaceBoundaryCondition* bc = (*iter);
            FaceBCOp* op = m_bc_init_op[bc->getId()];
            if (op)
              if (op->status())
                op->compute(bc);
          }
      }
  }
void ToyReactiveTransportModelService::updateBoundaryCondition()
  {
    if (m_bc_mng)
      {
        for (FaceBoundaryConditionIter iter = m_bc_mng->getBCIter(); iter.notEnd(); ++iter)
          {
            FaceBoundaryCondition* bc = (*iter);
            if (bc->isActive())
              {
                FaceBCOp* op = m_bc_update_op[bc->getId()];
                if (op)
                  if (op->status())
                    op->compute(bc);
              }
          }
      }
  }
void ToyReactiveTransportModelService::updateBoundaryCondition(
    ToyReactiveTransportModelService::FaceBoundaryCondition* bc)
  {
    debug(Trace::Medium) << "Update BC" << bc->getId();
    if (bc->isActive())
      {
        std::map<Integer, FaceBCOp*>::iterator iter = m_bc_update_op.find(
            bc->getId());
        if (iter != m_bc_update_op.end())
          {
            FaceBCOp* op = (*iter).second;
            if (op)
              if (op->status())
                op->compute(bc);
          }
      }
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_TOYREACTIVETRANSPORTMODEL(
    ToyReactiveTransportModel, ToyReactiveTransportModelService);
