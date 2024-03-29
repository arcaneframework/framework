﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef TOYREACTIVETRANSPORTMODELSERVICE_H
#define TOYREACTIVETRANSPORTMODELSERVICE_H
/* Author : willien at Tue Apr  1 14:52:11 2008
 * Generated by createNew
 */

#include "NumericalModel/Models/INumericalModel.h"

#include "Numerics/DiscreteOperator/IDivKGradDiscreteOperator.h"
#include "Numerics/LinearSolver/ILinearSolver.h"

#include "TimeUtils/ITimeStepMng.h"

#include "ToyReactiveTransportModel_axl.h"

#include "NumericalModel/Models/IIterativeTimeModel.h"
#include "NumericalModel/Models/ISubDomainModel.h"
#include "NumericalModel/Operators/INumericalModelVisitor.h"

#include "Mesh/Geometry/IGeometryMng.h"

#include <arcane/utils/FatalErrorException.h>

#include <map>

using namespace Arcane;

class ICollector;
class ITimeMng;
class INumericalModelVisitor;
template<typename SchemeType> class FluxModelT;
class IInterpolator;

class ToyReactiveTransportModelService: public ArcaneToyReactiveTransportModelObject,
    public IIterativeTimeModel,
    public ISubDomainModel
  {
public:

  typedef ISubDomainModel::FaceBoundaryConditionMng FaceBoundaryConditionMng;
  typedef FaceBoundaryConditionMng::BoundaryCondition FaceBoundaryCondition;
  typedef FaceBoundaryConditionMng::BoundaryConditionIter
      FaceBoundaryConditionIter;
  typedef FaceBoundaryCondition::BCOp<ToyReactiveTransportModelService>
      FaceBCOp;
  typedef ItemGroupMapT<Face, Real> BCValues;

  class Visitor: public INumericalModelVisitor
    {
  public:
    virtual ~Visitor()
      {
      }
    virtual String getName() = 0;
    virtual Integer visit(ToyReactiveTransportModelService* model)
      {
        throw Arcane::FatalErrorException(A_FUNCINFO, "not implemented");
      }
    virtual Integer visitForStart(ToyReactiveTransportModelService* model)
      {
        throw Arcane::FatalErrorException(A_FUNCINFO, "not implemented");
      }
    virtual Integer visitForFinalize(ToyReactiveTransportModelService* model)
      {
        throw Arcane::FatalErrorException(A_FUNCINFO, "not implemented");
      }
    };
  typedef std::map<Integer, ICollector*> CollectorList;
  typedef CollectorList::iterator CollectorListIter;
  typedef std::map<Integer, Visitor*> CollectorOpList;
  typedef CollectorOpList::iterator CollectorOpListIter;

  typedef FluxModelT<IDivKGradDiscreteOperator> DiffFluxModelType;
  typedef FluxModelT<IAdvectionOperator> AdvFluxModelType;

  /** Constructeur de la classe */
  ToyReactiveTransportModelService(const Arcane::ServiceBuildInfo & sbi) :
    ArcaneToyReactiveTransportModelObject(sbi), m_initialized(false), m_name(
        "ToyReactiveTransportModel"), m_numerical_domain(NULL), m_model_parent(
        NULL), m_geometry(NULL), m_diff_flux_scheme(NULL), m_diff_flux_model(
        NULL), m_adv_flux_scheme(NULL), m_adv_flux_model(NULL), m_bc_mng(NULL),
        m_time_step_mng(NULL), m_compute_velocity(false), m_error(0),
        m_output_level(0)
    {
    }

  /** Destructeur de la classe */
  virtual ~ToyReactiveTransportModelService()
    {
    }

  typedef ISubDomainModel::NumericalDomain NumericalDomain;

public:
  virtual String getName() const
    {
      return m_name;
    }
  void init();

  INumericalDomain* getINumericalDomain();
  NumericalDomain* getNumericalDomain();

  //! Reprise
  void continueInit()
    {
    }

  void startInit()
    {
    }
  //! start computation
  void start();
  void start(Integer sequence);

  void update()
    {
    }

  //! résolution du modèle
  Integer compute(Integer sequence);
  Integer baseCompute(Integer sequence);
  Integer baseCompute();
  template<typename Field>
  Integer computeT(Field& u_concentration, Field& v_concentration);

  void prepare(ICollector* collector, Integer sequence);
  void finalize(Integer sequence);
  void finalize()
    {
    }

  INumericalModel* getParent() const
    {
      return m_model_parent;
    }

  Integer getError()
    {
      return m_error;
    }
  void setParent(INumericalModel* parent)
    {
      m_model_parent = parent;
    }

  void startTimeStep();
  void endTimeStep()
    {
    }
  void setTimeMng(ITimeMng* time_mng)
    {
      m_time_mng = time_mng;
    }
  ITimeMng* getTimeMng()
    {
      return m_time_mng;
    }
  void setTimeStepMng(ITimeStepMng* time_step_mng)
    {
      m_time_step_mng = time_step_mng;
    }
  ITimeStepMng* getTimeStepMng()
    {
      return m_time_step_mng;
    }

  FaceBoundaryConditionMng* getFaceBoundaryConditionMng();
  void initBoundaryCondition();
  void initBoundaryCondition(FaceBoundaryCondition* bc);
  void updateBoundaryCondition();
  void updateBoundaryCondition(FaceBoundaryCondition* bc);

  IPostMng* getPostMng()
    {
      return NULL;
    }

  class Vars
    {
  public:
    Vars() :
      m_u_concentration(NULL)
      {
      }
    void setUConcentration(VariableCellReal* u_concentration)
      {
        m_u_concentration = u_concentration;
      }
    void setVConcentration(VariableCellReal* v_concentration)
      {
        m_v_concentration = v_concentration;
      }
    VariableCellReal* getUConcentration()
      {
        return m_u_concentration;
      }
    VariableCellReal* getVConcentration()
      {
        return m_v_concentration;
      }
  private:
    VariableCellReal* m_u_concentration;
    VariableCellReal* m_v_concentration;
    };
  void setVars(ICollector* collector);

  class TimeVisitorOp;
  Integer accept(INumericalModelVisitor* visitor);
  Integer acceptForStart(INumericalModelVisitor* visitor);
  Integer acceptForFinalize(INumericalModelVisitor* visitor);

  virtual void notifyNewSequence(Integer sequence)
    {
      SeqObsMapIter iter = m_obs.find(sequence);
      if (iter != m_obs.end())
        for (BufferT<INumericalModel::ISequenceObserver*>::iterator buf_iter =
            (*iter).second.begin(); buf_iter != (*iter).second.end(); ++buf_iter)
          (*buf_iter)->update();
    }
  virtual void addObs(INumericalModel::ISequenceObserver* obs, Integer sequence)
    {
      m_obs[sequence].add(obs);
    }

  typedef enum
    {
    TimeSeq, BaseSeq, AlgoSeq
    } eSequenceType;

  class Sequence
    {
  public:
    Sequence() :
      m_type(BaseSeq), m_op(NULL)
      {
      }
    Sequence(eSequenceType type, Visitor* op) :
      m_type(type), m_op(op)
      {
      }
    eSequenceType m_type;
    Visitor* m_op;
    };
  typedef std::map<Integer, Sequence> SequenceList;
private:
  typedef std::map<Integer, BufferT<INumericalModel::ISequenceObserver*> >
      SeqObsMap;
  typedef SeqObsMap::iterator SeqObsMapIter;
  SeqObsMap m_obs;
private:

  bool m_initialized;
  String m_name;

  // Expression Manager + Function parser
  IExpressionMng* m_expression_mng;
  IExpressionMng* m_local_expression_mng;
  FunctionParser m_rhs_function_parser;
  FunctionParser m_k_function_parser;
  FunctionParser m_psi_function_parser;

  NumericalDomain* m_numerical_domain;

  INumericalModel* m_model_parent;

  IGeometryMng* m_geometry;

  //!interpolator to compute cell_velocity from face_velocity
  IInterpolator* m_interpolator;
  IDivKGradDiscreteOperator* m_diff_flux_scheme;
  DiffFluxModelType* m_diff_flux_model;
  IAdvectionOperator* m_adv_flux_scheme;
  AdvFluxModelType* m_adv_flux_model;

  SubDomainModelProperty::eBoundaryConditionType m_interface_bc_type;

  FaceBoundaryConditionMng* m_bc_mng;
  Array<FaceBCOp*> m_bc_init_op;
  std::map<Integer, FaceBCOp*> m_bc_update_op;
  BCValues m_bc_values;
  BCValues m_bc_semi_values;

  Real m_current_time_step;
  Real m_current_time;
  ITimeMng* m_time_mng;
  ITimeStepMng* m_time_step_mng;

  CollectorList m_collectors;
  SequenceList m_sequences;
  Vars m_vars;

  bool m_compute_velocity;
  Integer m_error;

  Integer m_output_level;
  };

#endif /* TOYREACTIVETRANSPORTMODEL_H */
