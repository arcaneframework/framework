// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleSimpleHydroSimd.cc                                    (C) 2000-2020 */
/*                                                                           */
/* Simple Hydrodynamics Module with Vectorization.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#ifdef ARCANE_CHECK
#define ARCANE_TRACE_ENUMERATOR
#endif

#include "arcane/utils/List.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/EntryPoint.h"
#include "arcane/core/MathUtils.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/Concurrency.h"
#include "arcane/core/BasicService.h"
#include "arcane/core/ServiceBuildInfo.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/IMainFactory.h"

#include "arcane/tests/TypesSimpleHydro.h"

#include "arcane/core/SimdMathUtils.h"
#include "arcane/core/SimdItem.h"
#include "arcane/core/VariableView.h"

#ifdef __INTEL_COMPILER
#define HYDRO_PRAGMA_IVDEP _Pragma("ivdep")
#else
#define HYDRO_PRAGMA_IVDEP
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace SimpleHydro
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Simplified hydrodynamics module with vectorization and thread
 * parallelism.
 *
 * This module implements simple three-dimensional, parallel hydrodynamics
 * with cell-based pseudo-viscosity using the vectorization classes provided
 * by Arcane.
 */
class SimpleHydroSimdService
: public BasicService
, public ISimpleHydroService
{
 public:

  //! Constructor
  explicit SimpleHydroSimdService(const ServiceBuildInfo& sbi);
  ~SimpleHydroSimdService(); //!< Destructor

 public:

  virtual VersionInfo versionInfo() const { return VersionInfo(1, 0, 1); }

 public:

  void hydroBuild() override;
  void hydroStartInit() override;
  void hydroInit() override;
  void hydroExit() override;

  void computeForces() override;
  void computeVelocity() override;
  void computeViscosityWork() override;
  void applyBoundaryCondition() override;
  void moveNodes() override;
  void computeGeometricValues() override;
  void updateDensity() override;
  void applyEquationOfState() override;
  void computeDeltaT() override;

  void setModule(SimpleHydro::SimpleHydroModuleBase* module) override
  {
    m_module = module;
  }

 private:

  void computeGeometricValues2();

  void cellScalarPseudoViscosity();
  inline void computeCQs(Real3 node_coord[8], Real3 face_coord[6], Cell cell);

 private:

  VariableCellInt64 m_cell_unique_id; //!< Unique ID associated with the cell
  VariableCellInt32 m_sub_domain_id; //!< Sub-domain number associated with the cell
  VariableCellReal m_density; //!< Density per cell
  VariableCellReal m_pressure; //!< Pressure per cell
  VariableCellReal m_cell_mass; //!< Mass per cell
  VariableCellReal m_internal_energy; //!< Internal energy of the cells
  VariableCellReal m_volume; //!< Volume of the cells
  VariableCellReal m_old_volume; //!< Volume of a cell at the previous iteration
  VariableNodeReal3 m_force; //!< Force at the nodes
  VariableNodeReal3 m_velocity; //!< Velocity at the nodes
  VariableNodeReal m_node_mass; //! Node mass
  VariableCellReal m_cell_viscosity_force; //!< Local contribution of viscosity forces
  VariableCellReal m_viscosity_work; //!< Work done by viscosity forces per cell
  VariableCellReal m_adiabatic_cst; //!< Adiabatic constant per cell
  VariableCellReal m_caracteristic_length; //!< Characteristic length per cell
  VariableCellReal m_sound_speed; //!< Speed of sound in the cell
  VariableNodeReal3 m_node_coord; //!< Coordinates of the nodes
  VariableCellArrayReal3 m_cell_cqs; //!< Corner results for each cell

  VariableScalarReal m_density_ratio_maximum; //!< Maximum density increase over a time step
  VariableScalarReal m_delta_t_n; //!< Delta t n between t^{n-1/2} and t^{n+1/2}
  VariableScalarReal m_delta_t_f; //!< Delta t n+1/2 between t^{n} and t^{n+1}
  VariableScalarReal m_old_dt_f; //!< Delta t n-1/2 between t^{n-1} and t^{n}

  SimpleHydro::SimpleHydroModuleBase* m_module;

 private:

  void _computePressureAndCellPseudoViscosityForces();
  void _specialInit();

 public:

  void computeCQsSimd(SimdReal3 node_coord[8], SimdReal3 face_coord[6], SimdReal3 cqs[8]);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleHydroSimdService::
SimpleHydroSimdService(const ServiceBuildInfo& sbi)
: BasicService(sbi)
, m_cell_unique_id(VariableBuildInfo(sbi.mesh(), "UniqueId"))
, m_sub_domain_id(VariableBuildInfo(sbi.mesh(), "SubDomainId"))
, m_density(VariableBuildInfo(sbi.mesh(), "Density"))
, m_pressure(VariableBuildInfo(sbi.mesh(), "Pressure"))
, m_cell_mass(VariableBuildInfo(sbi.mesh(), "CellMass"))
, m_internal_energy(VariableBuildInfo(sbi.mesh(), "InternalEnergy"))
, m_volume(VariableBuildInfo(sbi.mesh(), "CellVolume"))
, m_old_volume(VariableBuildInfo(sbi.mesh(), "OldCellVolume"))
, m_force(VariableBuildInfo(sbi.mesh(), "Force", IVariable::PNoNeedSync))
, m_velocity(VariableBuildInfo(sbi.mesh(), "Velocity"))
, m_node_mass(VariableBuildInfo(sbi.mesh(), "NodeMass"))
, m_cell_viscosity_force(VariableBuildInfo(sbi.mesh(), "CellViscosityForce"))
, m_viscosity_work(VariableBuildInfo(sbi.mesh(), "ViscosityWork"))
, m_adiabatic_cst(VariableBuildInfo(sbi.mesh(), "AdiabaticCst"))
, m_caracteristic_length(VariableBuildInfo(sbi.mesh(), "CaracteristicLength"))
, m_sound_speed(VariableBuildInfo(sbi.mesh(), "SoundSpeed"))
, m_node_coord(VariableBuildInfo(sbi.mesh(), "NodeCoord"))
, m_cell_cqs(VariableBuildInfo(sbi.mesh(), "CellCQS"))
, m_density_ratio_maximum(VariableBuildInfo(sbi.mesh(), "DensityRatioMaximum"))
, m_delta_t_n(VariableBuildInfo(sbi.mesh(), "CenteredDeltaT"))
, m_delta_t_f(VariableBuildInfo(sbi.mesh(), "SplitDeltaT"))
, m_old_dt_f(VariableBuildInfo(sbi.mesh(), "OldDTf"))
, m_module(nullptr)
{
  if (TaskFactory::isActive()) {
    info() << "USE CONCURRENCY!!!!!";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleHydroSimdService::
~SimpleHydroSimdService()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHydroSimdService::
hydroBuild()
{
  info() << "Using hydro with vectorisation name=" << SimdInfo::name()
         << " vector_size=" << SimdReal::Length << " index_size=" << SimdInfo::Int32IndexSize;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHydroSimdService::
hydroExit()
{
  info() << "Hydro exit entry point";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Initialization of the hydro module when the case starts.
 */
void SimpleHydroSimdService::
hydroStartInit()
{
  info() << "START_INIT sizeof(ItemLocalId)=" << sizeof(ItemLocalId);
  ENUMERATE_CELL (icell, allCells()) {
    m_sub_domain_id[icell] = subDomain()->subDomainId();
    m_cell_unique_id[icell] = icell->uniqueId();
  }

  // Resize array variables
  m_cell_cqs.resize(8);

  //info() << "SimpleHydro SIMD initialisation vec_size=" << SimdReal::BLOCK_SIZE;

  // Check that initial values are correct
  {
    Integer nb_error = 0;
    auto in_pressure = viewIn(m_pressure);
    auto in_adiabatic_cst = viewIn(m_adiabatic_cst);
    VariableCellRealInView in_density = viewIn(m_density);
    ENUMERATE_CELL (icell, allCells()) {
      CellLocalId cid{ icell.asItemLocalId() };
      Real pressure = in_pressure[cid];
      Real adiabatic_cst = in_adiabatic_cst[cid];
      Real density = in_density[cid];
      if (math::isZero(pressure) || math::isZero(density) || math::isZero(adiabatic_cst)) {
        info() << "Null value for cell=" << ItemPrinter(*icell)
               << " density=" << density
               << " pressure=" << pressure
               << " adiabatic_cst=" << adiabatic_cst;
        ++nb_error;
      }
    }
    if (nb_error != 0)
      fatal() << "Some (" << nb_error << ") cells are not initialised";
  }

  // Initialize delta-t
  Real deltat_init = m_module->getDeltatInit();
  m_delta_t_n = deltat_init;
  m_delta_t_f = deltat_init;

  // Initialize geometric data: volume, cqs, characteristic lengths
  computeGeometricValues();

  m_node_mass.fill(ARCANE_REAL(0.0));
  m_velocity.fill(Real3::zero());

  // Initialization of cell masses and node masses
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    m_cell_mass[icell] = m_density[icell] * m_volume[icell];

    Real contrib_node_mass = ARCANE_REAL(0.125) * m_cell_mass[cell];
    for (NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node) {
      m_node_mass[i_node] += contrib_node_mass;
    }
  }

  m_node_mass.synchronize();

  // Initialize energy and sound speed
  auto in_pressure = viewIn(m_pressure);
  auto in_density = viewIn(m_density);
  auto in_adiabatic_cst = viewIn(m_adiabatic_cst);

  VariableCellRealOutView out_internal_energy = viewOut(m_internal_energy);
  auto out_sound_speed = viewOut(m_sound_speed);

  ENUMERATE_SIMD_CELL(icell, allCells())
  {
    SimdCell vi = *icell;
    SimdReal pressure = in_pressure[vi];
    SimdReal adiabatic_cst = in_adiabatic_cst[vi];
    SimdReal density = in_density[vi];
    out_internal_energy[vi] = pressure / ((adiabatic_cst - ARCANE_REAL(1.0)) * density);
    out_sound_speed[vi] = math::sqrt(adiabatic_cst * pressure / density);
  }
  info() << "END_START_INIT";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculation of forces at the current time \f$t^{n}\f$
 */
void SimpleHydroSimdService::
computeForces()
{
  // Resetting the force vector.
  m_force.fill(Real3::null());

  VariableNodeReal3OutView out_force = viewOut(m_force);
  VariableNodeReal3InView in_force = viewIn(m_force);

  // Calculation for each node of each cell the contribution
  // of pressure forces and pseudo-viscosity if necessary
  if (m_module->getViscosity() == TypesSimpleHydro::ViscosityCellScalar) {
    _computePressureAndCellPseudoViscosityForces();
  }
  else {
    ENUMERATE_CELL (icell, allCells()) {
      Cell cell = *icell;
      Real pressure = m_pressure[cell];
      for (NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node) {
        NodeLocalId nid = *i_node;
        out_force[nid] = in_force[nid] + pressure * m_cell_cqs[icell][i_node.index()];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Scalar pseudo-viscosity at the cells
 */
void SimpleHydroSimdService::
_computePressureAndCellPseudoViscosityForces()
{
  Real linear_coef = m_module->getViscosityLinearCoef();
  Real quadratic_coef = m_module->getViscosityQuadraticCoef();

  auto in_pressure = viewIn(m_pressure);
  auto in_density = viewIn(m_density);
  auto out_cell_viscosity_force = viewOut(m_cell_viscosity_force);

  // Loop over the cells of the mesh
  ENUMERATE_CELL (icell, allCells().view()) {
    Cell cell = *icell;
    CellLocalId cid(cell);

    const Real rho = in_density[cell];
    const Real pressure = in_pressure[icell];

    // Calculation of the velocity divergence
    Real delta_speed = 0.;
    for (NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node)
      delta_speed += math::scaMul(m_velocity[i_node], m_cell_cqs[icell][i_node.index()]);
    delta_speed /= m_volume[icell];

    // Capture only shocks
    bool shock = (math::min(ARCANE_REAL(0.0), delta_speed) < ARCANE_REAL(0.0));
    if (shock) {
      Real sound_speed = m_sound_speed[icell];
      Real dx = m_caracteristic_length[icell];
      Real quadratic_viscosity = rho * dx * dx * delta_speed * delta_speed;
      Real linear_viscosity = -rho * sound_speed * dx * delta_speed;
      Real scalar_viscosity = linear_coef * linear_viscosity + quadratic_coef * quadratic_viscosity;
      out_cell_viscosity_force[cid] = scalar_viscosity;
      for (NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node)
        m_force[i_node] += (pressure + scalar_viscosity) * m_cell_cqs[icell][i_node.index()];
    }
    else {
      out_cell_viscosity_force[cid] = ARCANE_REAL(0.0);
      for (NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node)
        m_force[i_node] += pressure * m_cell_cqs[icell][i_node.index()];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Momentum calculation (phase 2).
 */
void SimpleHydroSimdService::
computeVelocity()
{
  m_force.synchronize();

  auto in_old_velocity = viewIn(m_velocity);
  auto in_node_mass = viewIn(m_node_mass);
  auto in_force = viewIn(m_force);
  auto out_velocity = viewOut(m_velocity);
  // Calculate momentum at the nodes
  ENUMERATE_SIMD_NODE(i_node, allNodes())
  {
    SimdNode node = *i_node;
    SimdReal node_mass = in_node_mass[node];
    SimdReal3 old_velocity = in_old_velocity[node];
    SimdReal3 new_velocity = old_velocity + (m_delta_t_n() / node_mass) * in_force[i_node];
    out_velocity[node] = new_velocity;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Momentum calculation (phase 3).
 */
void SimpleHydroSimdService::
computeViscosityWork()
{
  arcaneParallelForeach(allCells(), [this](CellVectorView cells) {
    auto in_cell_viscosity_force = viewIn(m_cell_viscosity_force);
    auto out_viscosity_work = viewOut(m_viscosity_work);

    // Calculation of the work done by viscosity forces in a cell
    ENUMERATE_CELL (icell, cells) {
      Cell cell = *icell;
      CellLocalId cid(cell);
      Real work = 0.;
      Real scalar_viscosity = in_cell_viscosity_force[cid];
      if (!math::isZero(scalar_viscosity))
        for (NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node)
          work += math::scaMul(scalar_viscosity * m_cell_cqs[icell][i_node.index()], m_velocity[i_node]);
      out_viscosity_work[cid] = work;
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Accounting for boundary conditions.
 */
void SimpleHydroSimdService::
applyBoundaryCondition()
{
  for (auto bc : m_module->getBoundaryConditions()) {
    FaceGroup face_group = bc->getSurface();
    NodeGroup node_group = face_group.nodeGroup();
    Real value = bc->getValue();
    TypesSimpleHydro::eBoundaryCondition type = bc->getType();

    // loop over the faces of the surface
    ENUMERATE_FACE (j, face_group) {
      Face face = *j;
      Integer nb_node = face.nbNode();

      // loop over the nodes of the face
      for (Integer k = 0; k < nb_node; ++k) {
        Node node = face.node(k);
        switch (type) {
        case TypesSimpleHydro::VelocityX:
          m_velocity[node].x = value;
          break;
        case TypesSimpleHydro::VelocityY:
          m_velocity[node].y = value;
          break;
        case TypesSimpleHydro::VelocityZ:
          m_velocity[node].z = value;
          break;
        case TypesSimpleHydro::Unknown:
          break;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * \brief Moves the nodes.
 */
void SimpleHydroSimdService::
moveNodes()
{
  Real deltat_f = m_delta_t_f();

  ENUMERATE_NODE (i_node, allNodes()) {
    m_node_coord[i_node] += deltat_f * m_velocity[i_node];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Updates densities and calculates the maximum increase
 *	  of density across the entire mesh.
 */
void SimpleHydroSimdService::
updateDensity()
{
  Real density_ratio_maximum = ARCANE_REAL(0.0);

  HYDRO_PRAGMA_IVDEP
  ENUMERATE_CELL (icell, allCells()) {
    Real old_density = m_density[icell];
    Real new_density = m_cell_mass[icell] / m_volume[icell];

    m_density[icell] = new_density;

    Real density_ratio = (new_density - old_density) / new_density;

    if (density_ratio_maximum < density_ratio)
      density_ratio_maximum = density_ratio;
  }

  m_density_ratio_maximum = density_ratio_maximum;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies the equation of state and calculates the internal energy and
 * pressure.
 */
void SimpleHydroSimdService::
applyEquationOfState()
{
  const Real deltatf = m_delta_t_f();
  const bool add_viscosity_force = (m_module->getViscosity() != TypesSimpleHydro::ViscosityNo);

  auto in_adiabatic_cst = viewIn(m_adiabatic_cst);
  auto in_volume = viewIn(m_volume);
  auto in_density = viewIn(m_density);
  auto in_old_volume = viewIn(m_old_volume);
  auto in_internal_energy = viewIn(m_internal_energy);
  auto in_cell_mass = viewIn(m_cell_mass);
  auto in_viscosity_work = viewIn(m_viscosity_work);

  auto out_internal_energy = viewOut(m_internal_energy);
  auto out_sound_speed = viewOut(m_sound_speed);
  auto out_pressure = viewOut(m_pressure);

  // Calculation of internal energy
  arcaneParallelForeach(allCells(), [&](CellVectorView cells) {
    ENUMERATE_SIMD_CELL(icell, cells)
    {
      SimdCell vi = *icell;
      SimdReal adiabatic_cst = in_adiabatic_cst[vi];
      SimdReal volume_ratio = in_volume[vi] / in_old_volume[vi];
      SimdReal x = ARCANE_REAL(0.5) * (adiabatic_cst - ARCANE_REAL(1.0));
      SimdReal numer_accrois_nrj = ARCANE_REAL(1.0) + x * (ARCANE_REAL(1.0) - volume_ratio);
      SimdReal denom_accrois_nrj = ARCANE_REAL(1.0) + x * (ARCANE_REAL(1.0) - (ARCANE_REAL(1.0) / volume_ratio));
      SimdReal internal_energy = in_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj / denom_accrois_nrj);

      // Accounting for the work done by viscosity forces
      if (add_viscosity_force)
        internal_energy = internal_energy - deltatf * in_viscosity_work[vi] / (in_cell_mass[vi] * denom_accrois_nrj);

      out_internal_energy[vi] = internal_energy;

      SimdReal density = in_density[vi];
      SimdReal pressure = (adiabatic_cst - ARCANE_REAL(1.0)) * density * internal_energy;
      SimdReal sound_speed = math::sqrt(adiabatic_cst * pressure / density);
      out_pressure[vi] = pressure;
      out_sound_speed[vi] = sound_speed;
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the new time steps.
 */
void SimpleHydroSimdService::
computeDeltaT()
{
  const Real old_dt = m_global_deltat();

  // Calculation of the time step to respect the CFL criterion

  Real minimum_aux = FloatInfo<Real>::maxValue();
  Real new_dt = FloatInfo<Real>::maxValue();

  HYDRO_PRAGMA_IVDEP
  ENUMERATE_CELL (icell, ownCells()) {
    Real cell_dx = m_caracteristic_length[icell];
    Real sound_speed = m_sound_speed[icell];
    Real dx_sound = cell_dx / sound_speed;
    minimum_aux = math::min(minimum_aux, dx_sound);
  }

  new_dt = m_module->getCfl() * minimum_aux;

  // No excessively sudden variations up or down
  Real max_dt = (ARCANE_REAL(1.0) + m_module->getVariationSup()) * old_dt;
  Real min_dt = (ARCANE_REAL(1.0) - m_module->getVariationInf()) * old_dt;

  new_dt = math::min(new_dt, max_dt);
  new_dt = math::max(new_dt, min_dt);

  // Control of the relative increase of density
  Real dgr = m_module->getDensityGlobalRatio();
  if (m_density_ratio_maximum() > dgr)
    new_dt = math::min(old_dt * dgr / m_density_ratio_maximum(), new_dt);

  IParallelMng* pm = mesh()->parallelMng();
  new_dt = pm->reduce(Parallel::ReduceMin, new_dt);

  // Respecting the min and max values imposed by the .plt data file
  new_dt = math::min(new_dt, m_module->getDeltatMax());
  new_dt = math::max(new_dt, m_module->getDeltatMin());

  // The last calculation is done exactly at stopTime()
  {
    Real stop_time = m_module->getFinalTime();
    bool not_yet_finish = (m_global_time() < stop_time);
    bool too_much = ((m_global_time() + new_dt) > stop_time);

    if (not_yet_finish && too_much) {
      new_dt = stop_time - m_global_time();
      subDomain()->timeLoopMng()->stopComputeLoop(true);
    }
  }

  // Update variables
  m_old_dt_f.assign(old_dt);
  m_delta_t_n.assign(ARCANE_REAL(0.5) * (old_dt + new_dt));
  m_delta_t_f.assign(new_dt);
  m_global_deltat.assign(new_dt);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the results at the nodes of a hexahedral mesh.
 *
 * The method used is the four-triangle decomposition.
 */
inline void SimpleHydroSimdService::
computeCQsSimd(SimdReal3 node_coord[8], SimdReal3 face_coord[6], SimdReal3 cqs[8])
{
  const SimdReal3 c0 = face_coord[0];
  const SimdReal3 c1 = face_coord[1];
  const SimdReal3 c2 = face_coord[2];
  const SimdReal3 c3 = face_coord[3];
  const SimdReal3 c4 = face_coord[4];
  const SimdReal3 c5 = face_coord[5];

  const Real demi = ARCANE_REAL(0.5);
  const Real five = ARCANE_REAL(5.0);

  // Calculation of face 1 normals:
  const SimdReal3 n1a04 = demi * math::cross(node_coord[0] - c0, node_coord[3] - c0);
  const SimdReal3 n1a03 = demi * math::cross(node_coord[3] - c0, node_coord[2] - c0);
  const SimdReal3 n1a02 = demi * math::cross(node_coord[2] - c0, node_coord[1] - c0);
  const SimdReal3 n1a01 = demi * math::cross(node_coord[1] - c0, node_coord[0] - c0);

  // Calculation of face 2 normals:
  const SimdReal3 n2a05 = demi * math::cross(node_coord[0] - c1, node_coord[4] - c1);
  const SimdReal3 n2a12 = demi * math::cross(node_coord[4] - c1, node_coord[7] - c1);
  const SimdReal3 n2a08 = demi * math::cross(node_coord[7] - c1, node_coord[3] - c1);
  const SimdReal3 n2a04 = demi * math::cross(node_coord[3] - c1, node_coord[0] - c1);

  // Calculation of face 3 normals:
  const SimdReal3 n3a01 = demi * math::cross(node_coord[0] - c2, node_coord[1] - c2);
  const SimdReal3 n3a06 = demi * math::cross(node_coord[1] - c2, node_coord[5] - c2);
  const SimdReal3 n3a09 = demi * math::cross(node_coord[5] - c2, node_coord[4] - c2);
  const SimdReal3 n3a05 = demi * math::cross(node_coord[4] - c2, node_coord[0] - c2);

  // Calculation of face 4 normals:
  const SimdReal3 n4a09 = demi * math::cross(node_coord[4] - c3, node_coord[5] - c3);
  const SimdReal3 n4a10 = demi * math::cross(node_coord[5] - c3, node_coord[6] - c3);
  const SimdReal3 n4a11 = demi * math::cross(node_coord[6] - c3, node_coord[7] - c3);
  const SimdReal3 n4a12 = demi * math::cross(node_coord[7] - c3, node_coord[4] - c3);

  // Calculation of face 5 normals:
  const SimdReal3 n5a02 = demi * math::cross(node_coord[1] - c4, node_coord[2] - c4);
  const SimdReal3 n5a07 = demi * math::cross(node_coord[2] - c4, node_coord[6] - c4);
  const SimdReal3 n5a10 = demi * math::cross(node_coord[6] - c4, node_coord[5] - c4);
  const SimdReal3 n5a06 = demi * math::cross(node_coord[5] - c4, node_coord[1] - c4);

  // Calculation of face 6 normals:
  const SimdReal3 n6a03 = demi * math::cross(node_coord[2] - c5, node_coord[3] - c5);
  const SimdReal3 n6a08 = demi * math::cross(node_coord[3] - c5, node_coord[7] - c5);
  const SimdReal3 n6a11 = demi * math::cross(node_coord[7] - c5, node_coord[6] - c5);
  const SimdReal3 n6a07 = demi * math::cross(node_coord[6] - c5, node_coord[2] - c5);

  const Real real_1div12 = ARCANE_REAL(1.0) / ARCANE_REAL(12.0);

  cqs[0] = (five * (n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) +
            (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09)) *
  real_1div12;
  cqs[1] = (five * (n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) +
            (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07)) *
  real_1div12;
  cqs[2] = (five * (n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) +
            (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08)) *
  real_1div12;
  cqs[3] = (five * (n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) +
            (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11)) *
  real_1div12;
  cqs[4] = (five * (n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) +
            (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11)) *
  real_1div12;
  cqs[5] = (five * (n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) +
            (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02)) *
  real_1div12;
  cqs[6] = (five * (n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) +
            (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08)) *
  real_1div12;
  cqs[7] = (five * (n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) +
            (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03)) *
  real_1div12;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the volume of the meshes, characteristic lengths
 * and results at the vertices.
 */
void SimpleHydroSimdService::
computeGeometricValues()
{
  auto out_caracteristic_length = viewOut(m_caracteristic_length);
  arcaneParallelForeach(allCells(), [&](CellVectorView cells) {
    // Local copy of the cell vertex coordinates
    SimdReal3 coord[8];
    // Coordinates of the face centers
    SimdReal3 face_coord[6];

    SimdReal3 cqs[8];
    //std::cerr << "SIZE=" << cells.size() << '\n';
    ENUMERATE_SIMD_CELL(ivecitem, cells)
    {
      SimdCell vitem = *ivecitem;
      for (CellEnumerator iscell(ivecitem.enumerator()); iscell.hasNext(); ++iscell) {
        Cell cell(*iscell);
        Integer si(iscell.index());
        // Recopy the local coordinates (for the cache)
        for (NodeEnumerator i_node(cell.nodes()); i_node.index() < 8; ++i_node) {
          coord[i_node.index()].set(si, m_node_coord[i_node]);
          //info() << "COORD lid=" << cell.localId() << " i=" << i << " index=" << i_node.index() << " v=" << coord[i_node.index()][i];
        }
      }

      // Calculate the coordinates of the face centers
      face_coord[0] = ARCANE_REAL(0.25) * (coord[0] + coord[3] + coord[2] + coord[1]);
      face_coord[1] = ARCANE_REAL(0.25) * (coord[0] + coord[4] + coord[7] + coord[3]);
      face_coord[2] = ARCANE_REAL(0.25) * (coord[0] + coord[1] + coord[5] + coord[4]);
      face_coord[3] = ARCANE_REAL(0.25) * (coord[4] + coord[5] + coord[6] + coord[7]);
      face_coord[4] = ARCANE_REAL(0.25) * (coord[1] + coord[2] + coord[6] + coord[5]);
      face_coord[5] = ARCANE_REAL(0.25) * (coord[2] + coord[3] + coord[7] + coord[6]);

      // Calculate the characteristic length of the mesh.
      SimdReal3 median1 = face_coord[0] - face_coord[3];
      SimdReal3 median2 = face_coord[2] - face_coord[5];
      SimdReal3 median3 = face_coord[1] - face_coord[4];
      SimdReal d1 = math::normL2(median1);
      SimdReal d2 = math::normL2(median2);
      SimdReal d3 = math::normL2(median3);

      SimdReal dx_numerator = d1 * d2 * d3;
      SimdReal dx_denominator = d1 * d2 + d1 * d3 + d2 * d3;
      out_caracteristic_length[vitem] = dx_numerator / dx_denominator;

      //for( Integer i=0; i<NV; ++i ){
      //Cell cell(vitem.item(i));
      // Calculate the results at the vertices
      computeCQsSimd(coord, face_coord, cqs);
      //}
      // Calculate the results at the vertices:
      for (CellEnumerator si(ivecitem.enumerator()); si.hasNext(); ++si) {
        Cell cell(*si);
        Integer sidx(si.index());
        ArrayView<Real3> cqsv = m_cell_cqs[cell];
        for (Integer i_node = 0; i_node < 8; ++i_node)
          cqsv[i_node] = cqs[i_node][sidx];
      }

      /*for( Integer i=0; i<NV; ++i ){
          Cell cell(vitem.item(i));
          for( Integer z=0; z<8; ++z ){
            info() << "CQS lid=" << cell.localId() << " i=" << i << " z=" << z << " v=" << m_cell_cqs[cell][z];
          }
          }*/

      // Calculate the volume of the mesh
      for (CellEnumerator si(ivecitem.enumerator()); si.hasNext(); ++si) {
        Cell cell(*si);
        Integer sidx(si.index());
        Real volume = 0.;
        for (Integer i_node = 0; i_node < 8; ++i_node)
          volume += math::dot(coord[i_node][sidx], cqs[i_node][sidx]);
        volume /= ARCANE_REAL(3.0);

        m_old_volume[cell] = m_volume[cell];
        m_volume[cell] = volume;
      }
      /*SimdReal volume(0.0);
        for( Integer i_node=0; i_node<8; ++i_node )
          volume = volume + math::dot(coord[i_node],cqs[i_node]);
        volume = volume / ARCANE_REAL(3.0);

        for( Integer i=0; i<NV; ++i ){
          Cell cell(vitem.item(i));
          m_old_volume[cell] = m_volume[cell];
          m_volume[cell] = volume[i];
          }*/
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHydroSimdService::
hydroInit()
{
  if (m_module->getBackwardIteration() != 0) {
    subDomain()->timeLoopMng()->setBackwardSavePeriod(10);
  }
  info() << "INIT: DTmin=" << m_module->getDeltatMin()
         << " DTmax=" << m_module->getDeltatMax()
         << " DT=" << m_global_deltat();
  if (m_global_deltat() > m_module->getDeltatMax())
    ARCANE_FATAL("DeltaT > DTMax");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(SimpleHydroSimdService,
                        ServiceProperty("StdHydroSimdService", ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(SimpleHydro::ISimpleHydroService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace SimpleHydro

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
