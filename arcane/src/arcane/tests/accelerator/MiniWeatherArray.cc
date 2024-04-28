// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//////////////////////////////////////////////////////////////////////////////////////////

/*
** Copyright (c) 2018, National Center for Computational Sciences, Oak Ridge National Laboratory. All rights reserved.
**
** Portions Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
*/

// Copyright (c) 2018, National Center for Computational Sciences, Oak Ridge National Laboratory
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.

// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "arcane/utils/ITraceMng.h"
#include "arcane/BasicService.h"
#include "arcane/ServiceFactory.h"
#include "arcane/tests/MiniWeatherTypes.h"
#include "arcane/utils/NumArray.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/RunCommandLoop.h"

#include <math.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * NOTE: pour la version accélérateur, il est important de ne pas
 * avoir d'accès aux membres des classes dans les lambdas. Ces dernières
 * sont implicitements capturées par référence et donc lorsqu'on est
 * sur le GPU cela provoque un plantage car on essaie d'adresser la
 * mémoire où se trouve 'this'.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MiniWeatherArray
{
using namespace Arcane;
using namespace Arcane::Accelerator;
namespace ax = Arcane::Accelerator;

//const double pi = 3.14159265358979323846264338327;   //Pi
constexpr double grav = 9.8;                             //Gravitational acceleration (m / s^2)
constexpr double cp = 1004.;                             //Specific heat of dry air at constant pressure
constexpr double rd = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
constexpr double p0 = 1.e5;                              //Standard pressure at the surface in Pascals
constexpr double C0 = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
constexpr double gamm = 1.40027894002789400278940027894; //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
constexpr double xlen = 2.e4;     //Length of the domain in the x-direction (meters)
constexpr double zlen = 1.e4;     //Length of the domain in the z-direction (meters)
constexpr double hv_beta = 0.25;  //How strong to diffuse the solution: hv_beta \in [0:1]
constexpr double cfl = 1.50;      //"Courant, Friedrichs, Lewy" number (for numerical stability)
constexpr double max_speed = 450; //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
constexpr int hs = 2;             //"Halo" size: number of cells needed for a full "stencil" of information for reconstruction
constexpr int sten_size = 4;      //Size of the stencil used for interpolation

//Parameters for indexing and flags
constexpr int NUM_VARS = 4; //Number of fluid state variables
constexpr int ID_DENS = 0;  //index for density ("rho")
constexpr int ID_UMOM = 1;  //index for momentum in the x-direction ("rho * u")
constexpr int ID_WMOM = 2;  //index for momentum in the z-direction ("rho * w")
constexpr int ID_RHOT = 3;  //index for density * potential temperature ("rho * theta")
constexpr int DIR_X = 1;    //Integer constant to express that this operation is in the x-direction
constexpr int DIR_Z = 2;    //Integer constant to express that this operation is in the z-direction

//How is this not in the standard?!
inline double
dmin(double a, double b)
{
  if (a < b)
    return a;
  else
    return b;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MiniWeatherArrayBase
: public TraceAccessor
{
 public:
  MiniWeatherArrayBase(ITraceMng* tm) : TraceAccessor(tm){}
  virtual ~MiniWeatherArrayBase() = default;
 public:
  virtual int doOneIteration() =0;
  virtual void doExit(RealArrayView reduced_values) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LayoutType>
class MiniWeatherArray
: public MiniWeatherArrayBase
{
  using NumArray3Type = NumArray<double,MDDim3,LayoutType>;

 public:

  MiniWeatherArray(IAcceleratorMng* am,ITraceMng* tm,int nb_cell_x,int nb_cell_z,
                   double final_time,eMemoryRessource memory);
  
 public:
  
  int doOneIteration();
  void doExit(RealArrayView reduced_values);

 public:

  void init();
  static ARCCORE_HOST_DEVICE void injection(double x, double z, double &r, double &u,
                                            double &w, double &t, double &hr, double &ht);
  static ARCCORE_HOST_DEVICE void hydro_const_theta(double z, double &r, double &t);
  void output(NumArray3Type& state, double etime);
  void perform_timestep(NumArray3Type& state, NumArray3Type& state_tmp,
                        NumArray3Type& flux, NumArray3Type& tend, double dt);
  void semi_discrete_step(NumArray3Type& nstate_init, NumArray3Type& nstate_forcing,
                          NumArray3Type& nstate_out, double dt, int dir,
                          NumArray3Type& flux, NumArray3Type& tend);
  void compute_tendencies_x(NumArray3Type& nstate, NumArray3Type& flux);
  void compute_tendencies_final_x(NumArray3Type& flux, NumArray3Type& tend);
  void compute_tendencies_z(NumArray3Type& nstate, NumArray3Type& flux);
  void compute_tendencies_final_z(NumArray3Type& nstate, NumArray3Type& flux, NumArray3Type& tend);
  void set_halo_values_x(NumArray3Type& nstate);
  void set_halo_values_z(NumArray3Type& nstate);

 private:

  class ConstValues
  {
    ///////////////////////////////////////////////////////////////////////////////////////
    // Variables that are initialized but remain static over the course of the simulation
    ///////////////////////////////////////////////////////////////////////////////////////
   public:
    double sim_time;            //total simulation time in seconds
    double output_freq;         //frequency to perform output in seconds
    double dt;                  //Model time step (seconds)
    int nx, nz;                 //Number of local grid cells in the x- and z- dimensions
    int i_beg, k_beg;           // beginning index in the x- and z-directions
    int nranks, myrank;         // my rank id
    int left_rank, right_rank;  // Rank IDs that exist to my left and right in the global domain
    int nx_glob, nz_glob;       // Number of total grid cells in the x- and z- dimensions
    double dx, dz;              // Grid space length in x- and z-dimension (meters)
  };
  ConstValues m_const;
  double sim_time() const { return m_const.sim_time; }
  double output_freq() const { return m_const.output_freq; }
  double dt() const { return m_const.dt; }
  int nx() const { return m_const.nx; }
  int nz() const { return m_const.nz; }
  int i_beg() const { return m_const.i_beg; }
  int k_beg() const { return m_const.k_beg; }
  double dx() const { return m_const.dx; }
  double dz() const { return m_const.dz; }

  NumArray<double,MDDim1> hy_dens_cell;       // hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
  NumArray<double,MDDim1> hy_dens_theta_cell; // hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
  NumArray<double,MDDim1> hy_dens_int;        // hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
  NumArray<double,MDDim1> hy_dens_theta_int;  // hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
  NumArray<double,MDDim1> hy_pressure_int;    // hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

  ///////////////////////////////////////////////////////////////////////////////////////
  // Variables that are dynamics over the course of the simulation
  ///////////////////////////////////////////////////////////////////////////////////////
  double etime;          //Elapsed model time
  double output_counter; //Helps determine when it's time to do output
  // Runtime variable arrays
  NumArray3Type nstate;     // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  NumArray3Type nstate_tmp; // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  NumArray3Type nflux; // Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
  NumArray3Type ntend; // Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)
  int num_out = 0;   // The number of outputs performed so far
  int direction_switch = 1;
  ax::Runner* m_runner = nullptr;
  ax::RunQueue m_queue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LayoutType>
MiniWeatherArray<LayoutType>::
MiniWeatherArray(IAcceleratorMng* am, ITraceMng* tm, int nb_cell_x, int nb_cell_z,
                 double final_time, eMemoryRessource memory)
: MiniWeatherArrayBase(tm)
, hy_dens_cell(memory)
, hy_dens_theta_cell(memory)
, hy_dens_int(memory)
, hy_dens_theta_int(memory)
, nstate(memory)
, nstate_tmp(memory)
, nflux(memory)
, ntend(memory)
, m_runner(am->defaultRunner())
, m_queue(makeQueue(m_runner))
{
  m_const.nx_glob = nb_cell_x; // Number of total cells in the x-direction
  m_const.nz_glob = nb_cell_z; // Number of total cells in the z-direction
  m_const.dx = xlen / m_const.nx_glob;
  m_const.dz = zlen / m_const.nz_glob;

  info() << "Using 'MiniWeather' with accelerator";
  using Layout3Type = typename LayoutType::Layout3Type;
  auto layout_info = Layout3Type::layoutInfo();
  info() << "NumArrayLayout = " << layout_info[0] << " " << layout_info[1] << " " << layout_info[2];

  m_const.sim_time = final_time; //How many seconds to run the simulation
  m_const.output_freq = 100; //How frequently to output data to file (in seconds)
  //Set the cell grid size
  init();
  //Output the initial state
  output(nstate, etime);

  // Rend la file asynchrone
  // Ne le fait pas avec SYCL car cela provoque des résultats faux (avril 2024)
  if (m_queue.executionPolicy() != eExecutionPolicy::SYCL)
    m_queue.setAsync(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////

template<typename LayoutType>
int MiniWeatherArray<LayoutType>::
doOneIteration()
{
  ///////////////////////////////////////////////////////////////////////////////////////
  // BEGIN USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////
  //The x-direction length is twice as long as the z-direction length
  //So, you'll want to have nx_glob be twice as large as nz_glob
  //nx_glob = nb_cell_x;     //Number of total cells in the x-dirction
  //nz_glob = nb_cell_z;     //Number of total cells in the z-dirction
  //sim_time = 1500;   //How many seconds to run the simulation
  //sim_time = final_time;   //How many seconds to run the simulation
  //output_freq = 100; //How frequently to output data to file (in seconds)
  //output_freq = 20; //How frequently to output data to file (in seconds)
  ///////////////////////////////////////////////////////////////////////////////////////
  // END USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////

  //Output the initial state
  //output(state, etime);

  while (etime < m_const.sim_time)
  {
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt() > m_const.sim_time) {
      m_const.dt = m_const.sim_time - etime;
    }

    //Perform a single time step
    perform_timestep(nstate, nstate_tmp, nflux, ntend, dt());

    //Inform the user

    m_queue.barrier();
    info() << "Elapsed Time: " << etime << " / " << sim_time();

    //Update the elapsed time and output counter
    etime = etime + dt();
    output_counter = output_counter + dt();
    //If it's time for output, reset the counter, and do output

    if (output_counter >= output_freq()){
      output_counter = output_counter - output_freq();
      output(nstate, etime);
    }
    return 0;
  }
  return 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
template<typename LayoutType>
void MiniWeatherArray<LayoutType>::
perform_timestep(NumArray3Type& state, NumArray3Type& state_tmp,
                 NumArray3Type& flux, NumArray3Type& tend, double dt)
{
  if (direction_switch==1){
    //x-direction first
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_X, flux, tend);
    //z-direction second
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_Z, flux, tend);
  }
  else{
    //z-direction second
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_Z, flux, tend);
    //x-direction first
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_X, flux, tend);
  }
  if (direction_switch) {
    direction_switch = 0;
  }
  else
  {
    direction_switch = 1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
template<typename LayoutType>
void MiniWeatherArray<LayoutType>::
semi_discrete_step(NumArray3Type& nstate_init, NumArray3Type& nstate_forcing, NumArray3Type& nstate_out,
                   double dt, int dir, NumArray3Type& flux, NumArray3Type& tend)
{
  if (dir == DIR_X) {
    // Set the halo values  in the x-direction
    set_halo_values_x(nstate_forcing);
    // Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(nstate_forcing, flux);
    compute_tendencies_final_x(flux, tend);
  }
  else if (dir == DIR_Z){
    // Set the halo values  in the z-direction
    set_halo_values_z(nstate_forcing);
    // Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(nstate_forcing, flux);
    compute_tendencies_final_z(nstate_forcing, flux, tend);
  }

  auto command = makeCommand(m_queue);
  command.addKernelName("semi_discrete_step");
  auto state_init = ax::viewIn(command,nstate_init);
  auto state_out = ax::viewOut(command,nstate_out);
  auto in_tend = ax::viewIn(command,tend);

  command << RUNCOMMAND_LOOP3(iter,NUM_VARS,nz(),nx())
  {
    auto [ll, k, i] = iter();
    state_out(ll,k+hs,i+hs) = state_init(ll,k+hs,i+hs) + dt * in_tend(iter);
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Compute the time tendencies of the fluid state using forcing in the x-direction

//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
template<typename LayoutType>
void MiniWeatherArray<LayoutType>::
compute_tendencies_x(NumArray3Type& nstate, NumArray3Type& flux)
{
  const auto dx = this->dx();
  const auto nx = this->nx();
  const auto nz = this->nz();

  {
    auto command = makeCommand(m_queue);
    auto state = ax::viewIn(command,nstate);
    auto in_hy_dens_cell = ax::viewIn(command,hy_dens_cell);
    auto in_hy_dens_theta_cell = ax::viewIn(command,hy_dens_theta_cell);

    const double hv_coef = -hv_beta * dx / (16 * dt());
    //Compute fluxes in the x-direction for each cell
    auto out_flux = ax::viewOut(command,flux);

    command.addKernelName("compute_tendencies_x") << RUNCOMMAND_LOOP2(iter,nz,nx+1)
    {
      auto [k, i] = iter();
      double r, u, w, t, p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for ( int ll = 0; ll < NUM_VARS; ll++){
        for ( int s = 0; s < sten_size; s++)
          stencil[s] = state(ll,k+hs,i+s);

        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + in_hy_dens_cell(k + hs);
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = (vals[ID_RHOT] + in_hy_dens_theta_cell(k + hs)) / r;
      p = C0 * pow((r * t), gamm);

      // Compute the flux vector
      out_flux(ID_DENS,k,i) = r * u - hv_coef * d3_vals[ID_DENS];
      out_flux(ID_UMOM,k,i) = r * u * u + p - hv_coef * d3_vals[ID_UMOM];
      out_flux(ID_WMOM,k,i) = r * u * w - hv_coef * d3_vals[ID_WMOM];
      out_flux(ID_RHOT,k,i) = r * u * t - hv_coef * d3_vals[ID_RHOT];
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LayoutType> void MiniWeatherArray<LayoutType>::
compute_tendencies_final_x(NumArray3Type& flux, NumArray3Type& tend)
{
  const auto dx = this->dx();
  const auto nx = this->nx();
  const auto nz = this->nz();

  auto command = makeCommand(m_queue);
  auto in_flux = ax::viewIn(command,flux);
  auto out_tend = ax::viewOut(command,tend);
  // Use the fluxes to compute tendencies for each cell
  command.addKernelName("compute_tendencies_final_x") << RUNCOMMAND_LOOPN(iter,3,NUM_VARS,nz,nx)
  {
    auto [ll, k, i] = iter();
    out_tend(iter) = -(in_flux(ll,k,i+1) - in_flux(iter)) / dx;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Compute the time tendencies of the fluid state using forcing in the z-direction

//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
template<typename LayoutType> void MiniWeatherArray<LayoutType>::
compute_tendencies_z(NumArray3Type& nstate, NumArray3Type& flux)
{
  const auto dx = this->dx();
  const auto nx = this->nx();
  const auto nz = this->nz();

  {
    auto command = makeCommand(m_queue);
    auto in_hy_dens_int = ax::viewIn(command,hy_dens_int);
    auto in_hy_dens_theta_int = ax::viewIn(command,hy_dens_theta_int);
    auto in_hy_pressure_int = ax::viewIn(command,hy_pressure_int);

    auto state = ax::viewIn(command,nstate);
    //Compute the hyperviscosity coeficient
    const double hv_coef = -hv_beta * dx / (16 * dt());
    //Compute fluxes in the x-direction for each cell
    auto out_flux = ax::viewOut(command,flux);
    command.addKernelName("compute_tendencies_z") << RUNCOMMAND_LOOPN(iter,2,nz+1,nx)
    {
      auto [k,i] = iter();

      double r, u, w, t, p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (int ll = 0; ll < NUM_VARS; ll++){
        for (int s = 0; s < sten_size; s++)
          stencil[s] = state(ll,k+s,i+hs);

        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
        //First-order-accurate interpolation of the third spatial derivative of the state
        d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + in_hy_dens_int(k);
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = (vals[ID_RHOT] + in_hy_dens_theta_int(k)) / r;
      p = C0 * pow((r * t), gamm) - in_hy_pressure_int(k);

      //Compute the flux vector with hyperviscosity
      out_flux(ID_DENS,k,i) = r * w - hv_coef * d3_vals[ID_DENS];
      out_flux(ID_UMOM,k,i) = r * w * u - hv_coef * d3_vals[ID_UMOM];
      out_flux(ID_WMOM,k,i) = r * w * w + p - hv_coef * d3_vals[ID_WMOM];
      out_flux(ID_RHOT,k,i) = r * w * t - hv_coef * d3_vals[ID_RHOT];
    };
  }
}

template<typename LayoutType> void MiniWeatherArray<LayoutType>::
compute_tendencies_final_z(NumArray3Type& nstate, NumArray3Type& flux, NumArray3Type& tend)
{
  const auto dz = this->dz();
  const auto nx = this->nx();
  const auto nz = this->nz();

  auto command = makeCommand(m_queue);
  command.addKernelName("compute_tendencies_final_z");
  auto state = ax::viewIn(command,nstate);
  // Use the fluxes to compute tendencies for each cell
  auto in_flux = ax::viewIn(command,flux);
  auto out_tend = ax::viewOut(command,tend);
  const bool do_old = false;
  if (do_old){
    command << RUNCOMMAND_LOOP(iter,ArrayBounds<MDDim3>(NUM_VARS,nz,nx))
    {
      auto [ll, k, i] = iter();
      Real t = -(in_flux(ll,k+1,i) - in_flux(iter)) / dz;
      if (ll == ID_WMOM)
        t  = t - state(ID_DENS,k+hs,i+hs) * grav;
      out_tend(iter) = t;
    };
  }
  else{
    command << RUNCOMMAND_LOOP(iter,ArrayBounds<MDDim2>(nz,nx))
    {
      auto [k, i] = iter();
      double t1 = -(in_flux(ID_DENS,k+1,i) - in_flux(ID_DENS, k, i)) / dz;
      double t2 = -(in_flux(ID_UMOM,k+1,i) - in_flux(ID_UMOM, k, i)) / dz;
      double t3 = -(in_flux(ID_WMOM,k+1,i) - in_flux(ID_WMOM, k, i)) / dz;
      double t4 = -(in_flux(ID_RHOT,k+1,i) - in_flux(ID_RHOT, k, i)) / dz;

      //Real t = -(in_flux(ll,k+1,i) - in_flux(iter)) / dz;
      //if (ll == ID_WMOM)
      t3  = t3 - state(ID_DENS,k+hs,i+hs) * grav;
      //t  = t - state(ID_DENS,k+hs,i+hs) * grav;

      out_tend(ID_DENS, k, i) = t1;
      out_tend(ID_UMOM, k, i) = t2;
      out_tend(ID_WMOM, k, i) = t3;
      out_tend(ID_RHOT, k, i) = t4;
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LayoutType>
void MiniWeatherArray<LayoutType>::
set_halo_values_x(NumArray3Type& nstate)
{
  auto command = makeCommand(m_queue);

  auto state_in_out = ax::viewInOut(command, nstate);
  auto in_hy_dens_cell = ax::viewIn(command, hy_dens_cell);
  auto in_hy_dens_theta_cell = ax::viewIn(command, hy_dens_theta_cell);

  const auto nx = this->nx();
  const auto nz = this->nz();
  const auto dz = this->dz();
  const auto k_beg = this->k_beg();

  command << RUNCOMMAND_LOOP (iter, ArrayBounds<MDDim2>(NUM_VARS, nz))
  {
    auto [ll, k] = iter();
    state_in_out(ll, k + hs, 0) = state_in_out(ll, k + hs, nx + hs - 2);
    state_in_out(ll, k + hs, 1) = state_in_out(ll, k + hs, nx + hs - 1);
    state_in_out(ll, k + hs, nx + hs) = state_in_out(ll, k + hs, hs);
    state_in_out(ll, k + hs, nx + hs + 1) = state_in_out(ll, k + hs, hs + 1);
  };

  if (m_const.myrank == 0) {
    command << RUNCOMMAND_LOOP (iter, ArrayBounds<MDDim2>(nz, hs))
    {
      auto [k, i] = iter();
      double z = ((double)(k_beg + k) + 0.5) * dz;
      double v = z - 3 * zlen / 4;
      double compare_value = zlen / 16;
      // Normalement il faut comparer la valeur absolue via math::abs() mais
      // cette fonction n'est pas disponible avec DPC++ 2024.1 et le back-end CUDA
      // if (math::abs(v) <= compare_value) {
      if (v >= -compare_value && v <= compare_value) {
        state_in_out(ID_UMOM, k + hs, i) = (state_in_out(ID_DENS, k + hs, i) + in_hy_dens_cell(k + hs)) * 50.;
        state_in_out(ID_RHOT, k + hs, i) = (state_in_out(ID_DENS, k + hs, i) + in_hy_dens_cell(k + hs)) * 298. - in_hy_dens_theta_cell(k + hs);
      }
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Set this task's halo values in the z-direction.
//decomposition in the vertical direction.
template<typename LayoutType>
void MiniWeatherArray<LayoutType>::
set_halo_values_z(NumArray3Type& nstate)
{
  auto command = makeCommand(m_queue);

  const auto nx = this->nx();
  const auto nz = this->nz();

  auto state_in_out = ax::viewInOut(command,nstate);

  command << RUNCOMMAND_LOOP(iter,ArrayBounds<MDDim2>(NUM_VARS,nx+2*hs))
  {
    auto [ll,i] = iter();
    if (ll == ID_WMOM){
      state_in_out(ll,0,i) = 0.0;
      state_in_out(ll,1,i) = 0.0;
      state_in_out(ll,nz+hs,i) = 0.0;
      state_in_out(ll,nz+hs+1,i) = 0.0;
    }
    else {
      state_in_out(ll,0,i) = state_in_out(ll,hs,i);
      state_in_out(ll,1,i) = state_in_out(ll,hs,i); // GG: bizarre que ce soit pareil que au dessus.
      state_in_out(ll,nz+hs,i) = state_in_out(ll,nz+hs-1,i);
      state_in_out(ll,nz+hs+1,i) = state_in_out(ll,nz+hs-1,i); // Idem
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LayoutType>
void MiniWeatherArray<LayoutType>::
init()
{
  m_const.nranks = 1;
  m_const.myrank = 0;

  // For simpler version, replace i_beg = 0, nx = nx_glob, left_rank = 0, right_rank = 0;

  double nper = ((double)m_const.nx_glob) / m_const.nranks;
  m_const.i_beg = (int)(round(nper * (m_const.myrank)));
  int i_end = (int)(round(nper * ((m_const.myrank) + 1))) - 1;
  m_const.nx = i_end - m_const.i_beg + 1;
  m_const.left_rank = m_const.myrank - 1;
  if (m_const.left_rank == -1)
    m_const.left_rank = m_const.nranks - 1;
  m_const.right_rank = m_const.myrank + 1;
  if (m_const.right_rank == m_const.nranks)
    m_const.right_rank = 0;

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  m_const.k_beg = 0;
  m_const.nz = m_const.nz_glob;

  const auto dx = this->dx();
  const auto dz = this->dz();
  const auto nx = this->nx();
  const auto nz = this->nz();
  const auto i_beg = this->i_beg();
  const auto k_beg = this->k_beg();

  // Allocate the model data
  Int32 size0 = NUM_VARS * (nz + 2 * hs) * (nx + 2 * hs);
  info() << "Allocate memory for NumArray nx=" << nx << " nz=" << nz << " hs=" << hs << " size0=" << size0;
  nstate.resize(NUM_VARS,(nz + 2 * hs),(nx + 2 * hs));
  nstate_tmp.resize(NUM_VARS,(nz + 2 * hs),(nx + 2 * hs));
  nflux.resize(NUM_VARS,nz+1,nx+1); 
  ntend.resize(NUM_VARS,nz,nx);

  hy_dens_cell.resize(nz + 2 * hs);
  hy_dens_theta_cell.resize(nz + 2 * hs);
  hy_dens_int.resize(nz + 1);
  hy_dens_theta_int.resize(nz + 1);
  hy_pressure_int.resize(nz + 1);

  //Define the maximum stable time step based on an assumed maximum wind speed
  m_const.dt = dmin(dx, dz) / max_speed * cfl;
  //Set initial elapsed model time and output_counter to zero
  etime = 0.;
  output_counter = 0.;

  // Display grid information

  info() << "nx_glob, nz_glob: " << m_const.nx_glob << " " << m_const.nz_glob;
  info() << "dx,dz: " << dx << " " << dz;
  info() << "dt: " << dt();

  const int nqpoints = 3;
  const double qpoints[] =
  {
    0.112701665379258311482073460022E0,
    0.500000000000000000000000000000E0,
    0.887298334620741688517926539980E0
  };
  const double qweights[] =
  {
    0.277777777777777777777777777779E0,
    0.444444444444444444444444444444E0,
    0.277777777777777777777777777779E0
  };

  {
    auto command = makeCommand(m_queue);

    auto in_out_state = viewInOut(command,nstate);
    auto out_state_tmp = viewOut(command,nstate_tmp);

    //////////////////////////////////////////////////////////////////////////
    // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
    //////////////////////////////////////////////////////////////////////////

    command << RUNCOMMAND_LOOP(iter,ArrayBounds<MDDim2>(nz+2*hs,nx+2*hs))
    {
      auto [k,i] = iter();
      double r, u, w, t, hr, ht;
      for (int ll = 0; ll < NUM_VARS; ll++)
        in_out_state(ll,k,i) = 0.0;

      // Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (int kk = 0; kk < nqpoints; kk++) {
        for (int ii = 0; ii < nqpoints; ii++) {
          // Compute the x,z location within the global domain based on cell and quadrature index
          double x = ((double)(i_beg + i - hs) + 0.5) * dx + (qpoints[ii] - 0.5) * dx;
          double z = ((double)(k_beg + k - hs) + 0.5) * dz + (qpoints[kk] - 0.5) * dz;

          // Set the fluid state based on the user's specification (default is injection in this example)
          injection(x, z, r, u, w, t, hr, ht);

          // Store into the fluid state array
          in_out_state(ID_DENS,k,i) += + r * qweights[ii] * qweights[kk];
          in_out_state(ID_UMOM,k,i) += + (r + hr) * u * qweights[ii] * qweights[kk];
          in_out_state(ID_WMOM,k,i) += + (r + hr) * w * qweights[ii] * qweights[kk];
          in_out_state(ID_RHOT,k,i) += + ((r + hr) * (t + ht) - hr * ht) * qweights[ii] * qweights[kk];
        }
      }

      for (int ll = 0; ll < NUM_VARS; ll++)
        out_state_tmp(ll,k,i) = in_out_state(ll,k,i);
    };
    info() << "End init part 1\n";
  }

  // Compute the hydrostatic background state over vertical cell averages
  {
    auto command = makeCommand(m_queue);
    auto out_hy_dens_cell = viewOut(command,hy_dens_cell);
    auto out_hy_dens_theta_cell = viewOut(command,hy_dens_theta_cell);
    command << RUNCOMMAND_LOOP1(iter,(nz + 2 * hs)){
      auto [k] = iter();
      double r, u, w, t, hr, ht;
      double dens_cell = 0.0;
      double dens_theta_cell = 0.0;
      for (int kk = 0; kk < nqpoints; kk++){
        double z = (k_beg + (double)k - hs + 0.5) * dz;

        // Set the fluid state based on the user's specification (default is injection in this example)
        injection(0.0, z, r, u, w, t, hr, ht);

        dens_cell += hr * qweights[kk];
        dens_theta_cell += hr * ht * qweights[kk];
      }
      out_hy_dens_cell(k) = dens_cell;
      out_hy_dens_theta_cell(k) = dens_theta_cell;
    };
  }

  {
    auto command = makeCommand(m_queue);
    auto out_hy_dens_int = viewOut(command,hy_dens_int);
    auto out_hy_dens_theta_int = viewOut(command,hy_dens_theta_int);
    auto out_hy_pressure_int = viewOut(command,hy_pressure_int);
    command << RUNCOMMAND_LOOP1(iter,(nz + 1)){
      auto [k] = iter();
      double r, u, w, t, hr, ht;
      // Compute the hydrostatic background state at vertical cell interfaces
      double z = (k_beg + (double)k) * dz;

      //Set the fluid state based on the user's specification (default is injection in this example)
      injection(0.0, z, r, u, w, t, hr, ht);

      out_hy_dens_int(k) = hr;
      out_hy_dens_theta_int(k) = hr * ht;
      out_hy_pressure_int(k) = C0 * pow((hr * ht), gamm);
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// This test case is initially balanced but injects fast, cold air from the left boundary near the model top
// x and z are input coordinates at which to sample
// r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
// hr and ht are output background hydrostatic density and potential temperature at that location
template<typename LayoutType>
ARCCORE_HOST_DEVICE void MiniWeatherArray<LayoutType>::
injection(double x, double z, double &r, double &u, double &w, double &t, double &hr, double &ht)
{
  ARCANE_UNUSED(x);
  hydro_const_theta(z, hr, ht);
  r = 0.0;
  t = 0.0;
  u = 0.0;
  w = 0.0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
template<typename LayoutType>
ARCCORE_HOST_DEVICE void MiniWeatherArray<LayoutType>::
hydro_const_theta(double z, double &r, double &t)
{
  const double theta0 = 300.0; //Background potential temperature
  const double exner0 = 1.0;   //Surface-level Exner pressure
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                //Potential Temperature at z
  double exner = exner0 - grav * z / (cp * theta0); //Exner pressure at z
  double p = p0 * pow(exner, (cp / rd));            //Pressure at z
  double rt = pow((p / C0), (1. / gamm));           //rho*theta at z
  r = rt / t;                                //Density at z
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
//The file I/O uses netcdf, the only external library required for this mini-app.
//If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
template<typename LayoutType>
void MiniWeatherArray<LayoutType>::
output(NumArray3Type& state, double etime)
{
  ARCANE_UNUSED(state);
  ARCANE_UNUSED(etime);
  // Ne fait rien car on n'est pas branché avec 'NetCDF'.
}

// Affiche la somme sur les mailles des variables.
// Cela est utile pour la validation
template<typename LayoutType>
void MiniWeatherArray<LayoutType>::
doExit(RealArrayView reduced_values)
{
  int k, i, ll;
  double sum_v[NUM_VARS];

  // Comme le calcul se fait toujours sur l'hôte, il faut copier la valeur
  // de 'nstate' qui peut être sur le device.
  NumArray3Type host_nstate(eMemoryRessource::Host);
  host_nstate.copy(nstate);

  auto ns = host_nstate.constSpan();

  for (ll = 0; ll < NUM_VARS; ll++)
    sum_v[ll] = 0.0;
  for (k = 0; k < nz(); k++){
    for (i = 0; i < nx() + 1; i++){
      for (ll = 0; ll < NUM_VARS; ll++){
        sum_v[ll] += ns(ll,k+hs,i);
      }
    }
  }
  for ( int ll = 0; ll < NUM_VARS; ll++)
    reduced_values[ll] = sum_v[ll];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace ArcaneTest::MiniWeather;

class MiniWeatherArrayService
: public BasicService
, public IMiniWeatherService
{
 public:
  explicit MiniWeatherArrayService(const ServiceBuildInfo& sbi)
  : BasicService(sbi), m_p(nullptr){}
  ~MiniWeatherArrayService() override
  {
    delete m_p;
  }
 public:
  void init(IAcceleratorMng* am,Int32 nb_x,Int32 nb_z,Real final_time,
            eMemoryRessource r, bool use_left_layout) override
  {
    info() << "UseLeftLayout?=" << use_left_layout;
    if (use_left_layout)
      m_p = new MiniWeatherArray<LeftLayout>(am,traceMng(),nb_x,nb_z,final_time,r);
    else
      m_p = new MiniWeatherArray<RightLayout>(am,traceMng(),nb_x,nb_z,final_time,r);
  }
  bool loop() override
  {
    return m_p->doOneIteration()!=0;
  }
  void exit(RealArrayView reduced_values) override
  {
    m_p->doExit(reduced_values);
  }
 private:
  MiniWeatherArrayBase* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MiniWeatherArrayService,
                        ServiceProperty("MiniWeatherArray",ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IMiniWeatherService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MiniWeatherArray

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
