// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HllcSchemeModule.cc                                         (C) 2000-2026 */
/*                                                                           */
/* HLLC Scheme for Euler equations on unstructured 2D/3D mesh.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/BasicModule.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/EntryPoint.h"
#include "arcane/core/MathUtils.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/UnstructuredMeshConnectivity.h"

#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/tests/HllcSchemeTypes.h"
#include "arcane/tests/HllcScheme_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ax = Arcane::Accelerator;

namespace ArcaneTest::HllcScheme
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Module implementing the HLLC scheme for Euler equations.
 *
 * This module solves compressible 2D/3D Euler equations on unstructured mesh
 * using a finite volume scheme with:
 *   - HLLC Riemann solver (Einfeldt estimator)
 *   - MUSCL reconstruction with limiter (optional)
 *   - CFL time step on accelerator
 *   - Ideal gas law
 *
 * The main calculation loops use the Arcane accelerator API
 * (RUNCOMMAND_ENUMERATE, views, reducers).
 */
class HllcSchemeModule
: public ArcaneHllcSchemeObject
{
 public:

  explicit HllcSchemeModule(const ModuleBuildInfo& mb);
  ~HllcSchemeModule();

 public:

  static void staticInitialize(ISubDomain* sd);

 public:

  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 public:

  void build() override;
  void init() override;
  void computeLoop() override;
  void exit() override;

 public:

  void _computeDeltat();
  void _updateConservative();
  void _computePressure();

 private:

  void _computeGeometry();

  Real3 _faceNormal(const Face& face) const;
  Real _faceArea(const Real3& normal) const;
  Real3 _cellCenter(const Cell& cell) const;
  Real3 _faceCenter(const Face& face) const;

 private:

  VariableScalarReal m_global_deltat;

  Int32 m_dimension;

  Runner m_runner;
  RunQueue m_default_queue;
  UnstructuredMeshConnectivityView m_connectivity_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HllcSchemeModule::
HllcSchemeModule(const ModuleBuildInfo& mb)
: ArcaneHllcSchemeObject(mb)
, m_global_deltat(VariableBuildInfo(this, "GlobalDeltat"))
, m_dimension(3)
, m_runner(mb.subDomain()->acceleratorMng()->runner())
, m_default_queue(mb.subDomain()->acceleratorMng()->queue())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HllcSchemeModule::
~HllcSchemeModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HllcSchemeModule::
staticInitialize(ISubDomain* sd)
{
  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop("HllcSchemeLoop");
  {
    List<TimeLoopEntryPointInfo> build_list;
    build_list.add(TimeLoopEntryPointInfo("HllcScheme.HLLC_Build"));
    time_loop->setEntryPoints(ITimeLoop::WBuild, build_list);
  }
  {
    List<TimeLoopEntryPointInfo> init_list;
    init_list.add(TimeLoopEntryPointInfo("HllcScheme.HLLC_Init"));
    time_loop->setEntryPoints(ITimeLoop::WInit, init_list);
  }
  {
    List<TimeLoopEntryPointInfo> loop_list;
    loop_list.add(TimeLoopEntryPointInfo("HllcScheme.HLLC_ComputeLoop"));
    time_loop->setEntryPoints(String(ITimeLoop::WComputeLoop), loop_list);
  }
  {
    List<TimeLoopEntryPointInfo> exit_list;
    exit_list.add(TimeLoopEntryPointInfo("HllcScheme.HLLC_Exit"));
    time_loop->setEntryPoints(ITimeLoop::WExit, exit_list);
  }
  {
    StringList required;
    required.add("HllcScheme");
    required.add("ArcanePostProcessing");
    time_loop->setRequiredModulesName(required);
  }
  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HllcSchemeModule::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HllcSchemeModule::
init()
{
  m_dimension = mesh()->dimension();
  info() << "HLLC Scheme: initializing (dim=" << m_dimension << ")";

  if (m_dimension != 2 && m_dimension != 3)
    ARCANE_FATAL("Unsupported mesh dimension {0} (only 2 and 3 are supported)", m_dimension);

  // Initialise the connectivities for the accelerator
  m_connectivity_view.setMesh(mesh());

  _computeGeometry();

  // Initialize boundary conditions on faces
  {
    // Default value: all faces are at Wall (0)
    ENUMERATE_ (Face, iface, allFaces()) {
      m_face_bc_type[iface] = static_cast<Int32>(TypesHllcScheme::Wall);
    }

    for (Integer i = 0, nb = options()->boundaryCondition.count(); i < nb; ++i) {
      auto& bc = options()->boundaryCondition[i];
      FaceGroup face_group = bc.surface();
      TypesHllcScheme::eBoundaryCondition bc_type = bc.type();
      Real bc_density = bc.density();
      Real3 bc_velocity = Real3(bc.velocityX(), bc.velocityY(), bc.velocityZ());
      Real bc_pressure = bc.pressure();

      ENUMERATE_ (Face, iface, face_group) {
        m_face_bc_type[iface] = static_cast<Int32>(bc_type);
        m_face_bc_density[iface] = bc_density;
        m_face_bc_velocity[iface] = bc_velocity;
        m_face_bc_pressure[iface] = bc_pressure;
      }
    }
  }

  Real3 center = Real3::zero();
  {
    Real3 min_pos, max_pos;
    ENUMERATE_ (Node, inode, allNodes()) {
      Real3 pos = m_node_coord[inode];
      if (inode.index() == 0) {
        min_pos = pos;
        max_pos = pos;
        continue;
      }
      if (pos.x < min_pos.x)
        min_pos.x = pos.x;
      if (pos.y < min_pos.y)
        min_pos.y = pos.y;
      if (pos.z < min_pos.z)
        min_pos.z = pos.z;
      if (pos.x > max_pos.x)
        max_pos.x = pos.x;
      if (pos.y > max_pos.y)
        max_pos.y = pos.y;
      if (pos.z > max_pos.z)
        max_pos.z = pos.z;
    }
    center = Real(0.5) * (min_pos + max_pos);
    info() << "Mesh bounds: min=" << min_pos << " max=" << max_pos;
  }

  Real gamma = options()->gamma();

  ENUMERATE_ (Cell, icell, allCells()) {
    Real3 c = m_cell_center[icell];

    Real density_init, pressure_init;
    if (c.x < center.x) {
      density_init = Real(1.0);
      pressure_init = Real(1.0);
    }
    else {
      density_init = Real(0.125);
      pressure_init = Real(0.1);
    }

    m_density[icell] = density_init;
    m_pressure[icell] = pressure_init;
    m_momentum[icell] = Real3::zero();
    m_cell_mass[icell] = density_init * m_cell_volume[icell];
    m_energy[icell] = pressure_init / (gamma - Real(1.0));
    m_sound_speed[icell] = math::sqrt(gamma * pressure_init / density_init);
  }

  Real dt_init = options()->deltatInit();
  m_global_deltat = dt_init;

  info() << "HLLC Scheme: init complete (dim=" << m_dimension
         << ", gamma=" << gamma
         << ", cfl=" << options()->cfl()
         << ", order=" << options()->spatialOrder()
         << ", dt_init=" << dt_init << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the normal vector for a face (or edge in 2D).
 *
 * In 3D: normal = sum of half cross products of triangles
 * formed by the face center and each edge.
 * The resulting vector has a norm equal to the area of the face.
 *
 * In 2D: normal = perpendicular to the edge, norm = length of the edge.
 * Given an edge from p0 to p1: n = (p1.y-p0.y, p0.x-p1.x, 0).
 */
Real3 HllcSchemeModule::
_faceNormal(const Face& face) const
{
  NodeLocalIdView node_ids = face.nodeIds();
  Int32 nb_nodes = node_ids.size();

  if (m_dimension == 2) {
    Real3 p0 = m_node_coord[node_ids[0]];
    Real3 p1 = m_node_coord[node_ids[1]];
    Real3 edge = p1 - p0;
    return Real3(edge.y, -edge.x, Real(0.0));
  }

  Real3 normal = Real3::zero();
  Real3 fc = _faceCenter(face);

  for (Int32 i = 0; i < nb_nodes; ++i) {
    Int32 j = (i + 1) % nb_nodes;
    Real3 pi = m_node_coord[node_ids[i]];
    Real3 pj = m_node_coord[node_ids[j]];
    normal += Real(0.5) * math::cross(pi - fc, pj - fc);
  }

  return normal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real HllcSchemeModule::
_faceArea(const Real3& normal) const
{
  return normal.normL2();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 HllcSchemeModule::
_cellCenter(const Cell& cell) const
{
  Real3 center = Real3::zero();
  Int32 count = 0;
  for (NodeLocalId node_id : cell.nodeIds()) {
    center += m_node_coord[node_id];
    ++count;
  }
  if (count > 0)
    center /= Real(count);
  return center;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 HllcSchemeModule::
_faceCenter(const Face& face) const
{
  Real3 center = Real3::zero();
  Int32 count = 0;
  for (NodeLocalId node_id : face.nodeIds()) {
    center += m_node_coord[node_id];
    ++count;
  }
  if (count > 0)
    center /= Real(count);
  return center;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates volumes (areas in 2D), face normals, and centers.
 *
 * Volume/area calculated by the divergence theorem:
 *   3D: V = (1/3) * Σ_f (x_f · n_f)
 *   2D: A = (1/2) * Σ_e (x_e · n_e)
 *
 * This method remains on the CPU because it is only called once.
 */
void HllcSchemeModule::
_computeGeometry()
{
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    m_cell_center[icell] = _cellCenter(cell);
  }

  ENUMERATE_ (Face, iface, allFaces()) {
    Face face = *iface;
    Real3 normal = _faceNormal(face);
    m_face_normal[iface] = normal;
    m_face_area[iface] = _faceArea(normal);
    m_face_center[iface] = _faceCenter(face);
  }

  Real inv_dim = Real(1.0) / Real(m_dimension);

  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Real volume = Real(0.0);
    Real3 cell_center = m_cell_center[icell];

    for (Face face : cell.faces()) {
      Real3 normal = m_face_normal[face];
      Real3 fc = m_face_center[face];

      if (math::dot(fc - cell_center, normal) < Real(0.0))
        normal = -normal;

      volume += math::dot(fc, normal) * inv_dim;
    }

    if (volume <= Real(0.0))
      ARCANE_FATAL("Negative or zero cell volume ({0}) for cell {1}",
                   volume, cell.localId());
    m_cell_volume[icell] = volume;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the CFL time step.
 *
 * Characteristic length:
 *   3D: dx = V^(1/3)
 *   2D: dx = sqrt(A)
 */
void HllcSchemeModule::
_computeDeltat()
{
  Real gamma = options()->gamma();
  Real cfl = options()->cfl();
  Real inv_dim = Real(1.0) / Real(m_dimension);

  auto command = makeCommand(m_default_queue);

  auto in_density = ax::viewIn(command, m_density);
  auto in_momentum = ax::viewIn(command, m_momentum);
  auto in_pressure = ax::viewIn(command, m_pressure);
  auto in_cell_volume = ax::viewIn(command, m_cell_volume);

  ax::ReducerMin2<Real> min_dt_reducer(command);
  min_dt_reducer.setValue(FloatInfo<Real>::maxValue());

  command << RUNCOMMAND_ENUMERATE (Cell, cid, ownCells(), min_dt_reducer)
  {
    Real rho = in_density[cid];
    Real3 vel = in_momentum[cid] / rho;
    Real cs = math::sqrt(gamma * in_pressure[cid] / rho);
    Real speed = vel.normL2() + cs;

    Real dx = math::pow(in_cell_volume[cid], inv_dim);
    if (dx > FloatInfo<Real>::epsilon())
      min_dt_reducer.combine(cfl * dx / speed);
  };

  Real min_dt = min_dt_reducer.reducedValue();
  min_dt = parallelMng()->reduce(Parallel::ReduceMin, min_dt);

  Real old_dt = m_global_deltat();
  Real max_growth = Real(1.2);
  min_dt = math::min(min_dt, old_dt * max_growth);

  Real final_time = options()->finalTime();
  Real current_time = m_global_time();
  if (current_time + min_dt > final_time) {
    min_dt = final_time - current_time;
    subDomain()->timeLoopMng()->stopComputeLoop(true);
  }

  m_global_deltat = min_dt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Updates the conservative variables via HLLC fluxes.
 *
 * Uses a "cell-loop over faces" approach: each mesh iterates over its faces
 * and accumulates the flux contribution in local accumulators (no atomic
 * operation). Internal faces are treated twice (once by each adjacent mesh),
 * which is correct because the contribution is symmetric in sign.
 *
 * Slippery wall boundary conditions are handled directly in the face loop
 * when nbCell==1.
 */
void HllcSchemeModule::
_updateConservative()
{
  Real gamma = options()->gamma();
  Real dt = m_global_deltat();
  TypesHllcScheme::eLimiter limiter_type = options()->limiter();
  Int32 spatial_order = options()->spatialOrder();

  auto command = makeCommand(m_default_queue);

  // Connectivity accelerator
  auto fcc = m_connectivity_view.faceCell();
  auto cfc = m_connectivity_view.cellFace();

  // Input views (read-only)
  auto in_density = ax::viewIn(command, m_density);
  auto in_momentum = ax::viewIn(command, m_momentum);
  auto in_energy = ax::viewIn(command, m_energy);
  auto in_pressure = ax::viewIn(command, m_pressure);
  auto in_cell_volume = ax::viewIn(command, m_cell_volume);
  auto in_cell_center = ax::viewIn(command, m_cell_center);
  auto in_face_normal = ax::viewIn(command, m_face_normal);
  auto in_face_area = ax::viewIn(command, m_face_area);
  auto in_face_center = ax::viewIn(command, m_face_center);
  auto in_face_bc_type = ax::viewIn(command, m_face_bc_type);
  auto in_face_bc_density = ax::viewIn(command, m_face_bc_density);
  auto in_face_bc_velocity = ax::viewIn(command, m_face_bc_velocity);
  auto in_face_bc_pressure = ax::viewIn(command, m_face_bc_pressure);

  // Output views (write)
  auto out_density = ax::viewOut(command, m_density);
  auto out_momentum = ax::viewOut(command, m_momentum);
  auto out_energy = ax::viewOut(command, m_energy);

  command << RUNCOMMAND_ENUMERATE (Cell, cid, allCells())
  {
    Real flux_rho = Real(0.0);
    Real3 flux_mom = Real3::zero();
    Real flux_e = Real(0.0);

    // Iterate over the faces of cell cid
    for (FaceLocalId fid : cfc.faces(cid)) {
      auto cells = fcc.cells(fid);
      Int32 nb_face_cell = cells.size();

      Real3 normal = in_face_normal[fid];
      Real area = in_face_area[fid];
      Real inv_area = Real(1.0) / area;
      Real3 unit_normal = normal * inv_area;
      Real3 fc = in_face_center[fid];

      // -- Boundary face: boundary condition --
      if (nb_face_cell == 1) {
        Real3 c0 = in_cell_center[cid];
        if (math::dot(fc - c0, unit_normal) < Real(0.0))
          unit_normal = -unit_normal;

        Int32 bc_type = in_face_bc_type[fid];

        PrimitiveState p_cell;
        p_cell.density = in_density[cid];
        p_cell.velocity = in_momentum[cid] / p_cell.density;
        p_cell.pressure = in_pressure[cid];

        PrimitiveState p_bc;
        FluxState f;

        if (bc_type == static_cast<Int32>(TypesHllcScheme::Outflow)) {
          // Free outflow (zero gradient): use the cell state
          p_bc = p_cell;
          f = eulerFlux(p_bc, unit_normal, gamma);
        }
        else if (bc_type == static_cast<Int32>(TypesHllcScheme::Inflow)) {
          // Imposed inflow: use the specified state
          p_bc.density = in_face_bc_density[fid];
          p_bc.velocity = in_face_bc_velocity[fid];
          p_bc.pressure = in_face_bc_pressure[fid];
          f = eulerFlux(p_bc, unit_normal, gamma);
        }
        else {
          // Slippery wall (default): mirror of the normal velocity
          Real vn = math::dot(p_cell.velocity, unit_normal);
          if (vn < Real(0.0)) {
            p_bc = p_cell;
            p_bc.velocity = p_cell.velocity - Real(2.0) * vn * unit_normal;
            f = eulerFlux(p_bc, unit_normal, gamma);
          }
          else {
            f.density_flux = Real(0.0);
            f.momentum_flux = Real3::zero();
            f.energy_flux = Real(0.0);
          }
        }

        flux_rho += f.density_flux * area;
        flux_mom += f.momentum_flux * area;
        flux_e += f.energy_flux * area;
        continue;
      }

      // -- Internal face: two adjacent cells --
      CellLocalId cl = cells[0];
      CellLocalId cr = cells[1];

      // Primitive states at cell centers
      PrimitiveState pL, pR;
      {
        Real rl = in_density[cl];
        Real rr = in_density[cr];
        pL.density = rl;
        pL.velocity = in_momentum[cl] / rl;
        pL.pressure = in_pressure[cl];
        pR.density = rr;
        pR.velocity = in_momentum[cr] / rr;
        pR.pressure = in_pressure[cr];
      }

      // Orient the normal to point from cl to cr
      Real3 diff_dir = in_cell_center[cr] - in_cell_center[cl];
      bool swapped = false;
      if (math::dot(diff_dir, unit_normal) < Real(0.0)) {
        unit_normal = -unit_normal;
        swapped = true;
      }

      // Adjust the pair (pL,pR) if necessary so that
      // pL corresponds to the "left side" of the normal
      if (swapped) {
        PrimitiveState tmp = pL;
        pL = pR;
        pR = tmp;
      }

      // Second-order MUSCL reconstruction
      if (spatial_order >= 2) {
        Real3 dx_l = fc - in_cell_center[cl];
        Real3 dx_r = fc - in_cell_center[cr];
        Real dx_n_l = math::dot(dx_l, unit_normal);
        Real dx_n_r = math::dot(dx_r, unit_normal);

        // Simplified limiter: r = |dq|/(|dq|+eps)
        auto recon_scalar = [&](Real qL, Real qR, Real dx) -> Real {
          Real dq = qR - qL;
          Real eps = FloatInfo<Real>::epsilon();
          Real adq = math::abs(dq);
          Real r = (adq > eps) ? adq / (adq + eps) : Real(1.0);
          Real phi = limiter(limiter_type, r);
          Real val = qL + Real(0.5) * phi * dx * dq;
          return (val > eps) ? val : eps;
        };

        pL.density = recon_scalar(pL.density, pR.density, dx_n_l);
        pR.density = recon_scalar(pR.density, pL.density, dx_n_r);
        pL.pressure = recon_scalar(pL.pressure, pR.pressure, dx_n_l);
        pR.pressure = recon_scalar(pR.pressure, pL.pressure, dx_n_r);

        Real dvn = math::dot(pR.velocity - pL.velocity, unit_normal);
        Real eps = FloatInfo<Real>::epsilon();
        Real advn = math::abs(dvn);
        Real r_v = (advn > eps) ? advn / (advn + eps) : Real(1.0);
        Real phi_v = limiter(limiter_type, r_v);

        pL.velocity += Real(0.5) * phi_v * dx_n_l * dvn * unit_normal;
        pR.velocity += Real(0.5) * phi_v * dx_n_r * dvn * unit_normal;
      }

      // HLLC flux at the face
      FluxState flux = hllcFlux(pL, pR, unit_normal, gamma);
      flux.density_flux *= area;
      flux.momentum_flux *= area;
      flux.energy_flux *= area;

      // Contribution to the cumulative flux for cell cid:
      //   If cid == cl (left side) -> + sign (outgoing flux)
      //   If cid == cr (right side) -> - sign (incoming flux)
      Real side = Real(1.0);
      if (cid == cl)
        side = Real(1.0);
      else
        side = Real(-1.0);

      flux_rho += side * flux.density_flux;
      flux_mom += side * flux.momentum_flux;
      flux_e += side * flux.energy_flux;
    }

    // Conservative update: U^{n+1} = U^n - dt/V * Σ(F·n·A)
    Real inv_vol = Real(1.0) / in_cell_volume[cid];
    Real new_density = in_density[cid] - dt * inv_vol * flux_rho;
    Real3 new_momentum = in_momentum[cid] - dt * inv_vol * flux_mom;
    Real new_energy = in_energy[cid] - dt * inv_vol * flux_e;

    if (new_density <= Real(0.0))
      new_density = FloatInfo<Real>::epsilon();
    if (new_energy <= Real(0.0))
      new_energy = FloatInfo<Real>::epsilon();

    out_density[cid] = new_density;
    out_momentum[cid] = new_momentum;
    out_energy[cid] = new_energy;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the pressure using the ideal gas law (on accelerator).
 *
 *   p = (γ-1) * (ρE - ½|ρu|²/ρ)
 */
void HllcSchemeModule::
_computePressure()
{
  Real gamma = options()->gamma();

  auto command = makeCommand(m_default_queue);

  auto in_density = ax::viewIn(command, m_density);
  auto in_momentum = ax::viewIn(command, m_momentum);
  auto in_energy = ax::viewIn(command, m_energy);
  auto out_pressure = ax::viewOut(command, m_pressure);
  auto out_sound_speed = ax::viewOut(command, m_sound_speed);

  command << RUNCOMMAND_ENUMERATE (Cell, cid, allCells())
  {
    Real rho = in_density[cid];
    Real3 rho_u = in_momentum[cid];
    Real rho_E = in_energy[cid];

    Real rho_ek = Real(0.5) * math::dot(rho_u, rho_u) / rho;
    Real pressure = (gamma - Real(1.0)) * (rho_E - rho_ek);

    if (pressure <= Real(0.0))
      pressure = FloatInfo<Real>::epsilon();

    out_pressure[cid] = pressure;
    out_sound_speed[cid] = math::sqrt(gamma * pressure / rho);
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HllcSchemeModule::
computeLoop()
{
  _computeDeltat();

  Real dt = m_global_deltat();
  Real current_time = m_global_time();

  info() << "HLLC step: t=" << current_time << " dt=" << dt
         << " iter=" << m_global_iteration();

  _updateConservative();
  _computePressure();

  m_global_time.assign(current_time + dt);

  if (m_global_time() >= options()->finalTime())
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HllcSchemeModule::
exit()
{
  info() << "HLLC Scheme: final t=" << m_global_time()
         << " iterations=" << m_global_iteration();

  Real sum_mass = Real(0.0);
  Real sum_energy = Real(0.0);
  Real sum_pressure = Real(0.0);

  ENUMERATE_ (Cell, icell, ownCells()) {
    sum_mass += m_density[icell];
    sum_energy += m_energy[icell];
    sum_pressure += m_pressure[icell];
  }

  sum_mass = parallelMng()->reduce(Parallel::ReduceSum, sum_mass);
  sum_energy = parallelMng()->reduce(Parallel::ReduceSum, sum_energy);
  sum_pressure = parallelMng()->reduce(Parallel::ReduceSum, sum_pressure);

  info() << "HLLC sums: density=" << sum_mass
         << " energy=" << sum_energy
         << " pressure=" << sum_pressure;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(HllcSchemeModule, HllcScheme);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest::HllcScheme

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
