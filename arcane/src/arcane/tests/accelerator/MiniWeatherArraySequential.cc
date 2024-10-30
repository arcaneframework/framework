// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
#include "arcane/utils/NumArray.h"

#include "arcane/BasicService.h"
#include "arcane/ServiceFactory.h"

#include "arcane/tests/MiniWeatherTypes.h"

#include <math.h>

namespace MiniWeatherArraySequential
{
using namespace Arcane;
using namespace Arcane::Accelerator;

//const double pi = 3.14159265358979323846264338327;   //Pi
const double grav = 9.8;                             //Gravitational acceleration (m / s^2)
const double cp = 1004.;                             //Specific heat of dry air at constant pressure
const double rd = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
const double p0 = 1.e5;                              //Standard pressure at the surface in Pascals
const double C0 = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
const double gamm = 1.40027894002789400278940027894; //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
const double xlen = 2.e4;     //Length of the domain in the x-direction (meters)
const double zlen = 1.e4;     //Length of the domain in the z-direction (meters)
const double hv_beta = 0.25;  //How strong to diffuse the solution: hv_beta \in [0:1]
const double cfl = 1.50;      //"Courant, Friedrichs, Lewy" number (for numerical stability)
const double max_speed = 450; //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const int hs = 2;             //"Halo" size: number of cells needed for a full "stencil" of information for reconstruction
const int sten_size = 4;      //Size of the stencil used for interpolation

//Parameters for indexing and flags
const int NUM_VARS = 4; //Number of fluid state variables
const int ID_DENS = 0;  //index for density ("rho")
const int ID_UMOM = 1;  //index for momentum in the x-direction ("rho * u")
const int ID_WMOM = 2;  //index for momentum in the z-direction ("rho * w")
const int ID_RHOT = 3;  //index for density * potential temperature ("rho * theta")
const int DIR_X = 1;    //Integer constant to express that this operation is in the x-direction
const int DIR_Z = 2;    //Integer constant to express that this operation is in the z-direction

const int nqpoints = 3;
double qpoints[] = {0.112701665379258311482073460022E0, 0.500000000000000000000000000000E0, 0.887298334620741688517926539980E0};
double qweights[] = {0.277777777777777777777777777779E0, 0.444444444444444444444444444444E0, 0.277777777777777777777777777779E0};

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

class MiniWeatherArraySequential
: public TraceAccessor
{
 public:

  MiniWeatherArraySequential(ITraceMng* tm,int nb_cell_x,int nb_cell_z,double final_time);
  
 public:
  
  int doOneIteration();
  void doExit(RealArrayView reduced_values);

 private:

  void init();
  void injection(double x, double z, double &r, double &u, double &w, double &t, double &hr, double &ht);
  void hydro_const_theta(double z, double &r, double &t);
  void output(NumArray<double,MDDim3>& state, double etime);
  void perform_timestep(NumArray<double,MDDim3>& state, NumArray<double,MDDim3>& state_tmp,
                        NumArray<double,MDDim3>& flux, NumArray<double,MDDim3>& tend, double dt);
  void semi_discrete_step(NumArray<double,MDDim3>& nstate_init, NumArray<double,MDDim3>& nstate_forcing,
                          NumArray<double,MDDim3>& nstate_out, double dt, int dir, NumArray<double,MDDim3>& flux,
                          NumArray<double,MDDim3>& tend);
  void compute_tendencies_x(NumArray<double,MDDim3>& nstate, NumArray<double,MDDim3>& flux, NumArray<double,MDDim3>& tend);
  void compute_tendencies_z(NumArray<double,MDDim3>& nstate, NumArray<double,MDDim3>& flux, NumArray<double,MDDim3>& tend);
  void set_halo_values_x(NumArray<double,MDDim3>& nstate);
  void set_halo_values_z(NumArray<double,MDDim3>& nstate);

 private:

  const int nx_glob, nz_glob;       // Number of total grid cells in the x- and z- dimensions
  const double dx, dz;              // Grid space length in x- and z-dimension (meters)

  ///////////////////////////////////////////////////////////////////////////////////////
  // Variables that are initialized but remain static over the course of the simulation
  ///////////////////////////////////////////////////////////////////////////////////////
  double sim_time;            //total simulation time in seconds
  double output_freq;         //frequency to perform output in seconds
  double dt;                  //Model time step (seconds)
  int nx, nz;                 //Number of local grid cells in the x- and z- dimensions
  int i_beg, k_beg;           // beginning index in the x- and z-directions
  int nranks, myrank;         // my rank id
  int left_rank, right_rank;  // Rank IDs that exist to my left and right in the global domain
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
  NumArray<double,MDDim3> nstate;     // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  NumArray<double,MDDim3> nstate_tmp; // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  NumArray<double,MDDim3> nflux; // Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
  NumArray<double,MDDim3> ntend; // Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)
  int direction_switch = 1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MiniWeatherArraySequential::
MiniWeatherArraySequential(ITraceMng* tm,int nb_cell_x,int nb_cell_z,double final_time)
: TraceAccessor(tm)
, nx_glob(nb_cell_x)     //Number of total cells in the x-dirction
, nz_glob(nb_cell_z)     //Number of total cells in the z-dirction
, dx(xlen / nx_glob)
, dz(zlen / nz_glob)
{
  sim_time = final_time;   //How many seconds to run the simulation
  output_freq = 100; //How frequently to output data to file (in seconds)
  //Set the cell grid size
  init();
  //Output the initial state
  output(nstate, etime);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////

int MiniWeatherArraySequential::
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

  while (etime < sim_time)
  {
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt > sim_time) {
      dt = sim_time - etime;
    }

    //Perform a single time step
    perform_timestep(nstate, nstate_tmp, nflux, ntend, dt);

    //Inform the user

    info() << "Elapsed Time: " << etime << " / " << sim_time;

    //Update the elapsed time and output counter
    etime = etime + dt;
    output_counter = output_counter + dt;
    //If it's time for output, reset the counter, and do output

    if (output_counter >= output_freq){
      output_counter = output_counter - output_freq;
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
void MiniWeatherArraySequential::
perform_timestep(NumArray<double,MDDim3>& state, NumArray<double,MDDim3>& state_tmp,
                 NumArray<double,MDDim3>& flux, NumArray<double,MDDim3>& tend, double dt)
{
  if (direction_switch){
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
void MiniWeatherArraySequential::
semi_discrete_step(NumArray<double,MDDim3>& nstate_init, NumArray<double,MDDim3>& nstate_forcing, NumArray<double,MDDim3>& nstate_out,
                   double dt, int dir, NumArray<double,MDDim3>& flux, NumArray<double,MDDim3>& tend)
{
  auto state_init = nstate_init.constMDSpan();
  auto state_out = nstate_out.mdspan();
  auto in_tend = tend.constMDSpan();

  int i, k, ll;
  if (dir == DIR_X) {
    // Set the halo values  in the x-direction
    set_halo_values_x(nstate_forcing);
    // Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(nstate_forcing, flux, tend);
  }
  else if (dir == DIR_Z){
    // Set the halo values  in the z-direction
    set_halo_values_z(nstate_forcing);
    // Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(nstate_forcing, flux, tend);
  }

  // Apply the tendencies to the fluid state
  for (ll = 0; ll < NUM_VARS; ll++){
    for (k = 0; k < nz; k++){
      for (i = 0; i < nx; i++){
        //indt = ll * nz * nx + k * nx + i;
        state_out(ll,k+hs,i+hs) = state_init(ll,k+hs,i+hs) + dt * in_tend(ll,k,i);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Compute the time tendencies of the fluid state using forcing in the x-direction

//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void MiniWeatherArraySequential::
compute_tendencies_x(NumArray<double,MDDim3>& nstate, NumArray<double,MDDim3>& flux, NumArray<double,MDDim3>& tend)
{
  auto state = nstate.constMDSpan();

  int i, k, ll, s;
  double r, u, w, t, p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coeficient
  hv_coef = -hv_beta * dx / (16 * dt);
  //Compute fluxes in the x-direction for each cell
  auto out_flux = flux.mdspan();
  for (k = 0; k < nz; k++){
    for (i = 0; i < nx + 1; i++){
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll = 0; ll < NUM_VARS; ll++){
        for (s = 0; s < sten_size; s++)
          stencil[s] = state(ll,k+hs,i+s);

        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_cell(k + hs);
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = (vals[ID_RHOT] + hy_dens_theta_cell(k + hs)) / r;
      p = C0 * pow((r * t), gamm);

      // Compute the flux vector
      out_flux(ID_DENS,k,i) = r * u - hv_coef * d3_vals[ID_DENS];
      out_flux(ID_UMOM,k,i) = r * u * u + p - hv_coef * d3_vals[ID_UMOM];
      out_flux(ID_WMOM,k,i) = r * u * w - hv_coef * d3_vals[ID_WMOM];
      out_flux(ID_RHOT,k,i) = r * u * t - hv_coef * d3_vals[ID_RHOT];
    }
  }

  auto in_flux = flux.constMDSpan();
  auto out_tend = tend.mdspan();
  // Use the fluxes to compute tendencies for each cell
  for (ll = 0; ll < NUM_VARS; ll++) {
    for (k = 0; k < nz; k++) {
      for (i = 0; i < nx; i++) {
        out_tend(ll,k,i) = -(in_flux(ll,k,i+1) - in_flux(ll,k,i)) / dx;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Compute the time tendencies of the fluid state using forcing in the z-direction

//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void MiniWeatherArraySequential::
compute_tendencies_z(NumArray<double,MDDim3>& nstate, NumArray<double,MDDim3>& flux, NumArray<double,MDDim3>& tend)
{
  auto state = nstate.constMDSpan();
  double r, u, w, t, p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coeficient
  hv_coef = -hv_beta * dx / (16 * dt);
  //Compute fluxes in the x-direction for each cell
  auto out_flux = flux.mdspan();
  for (int k = 0; k < nz + 1; k++){
    for (int i = 0; i < nx; i++){
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
      r = vals[ID_DENS] + hy_dens_int(k);
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = (vals[ID_RHOT] + hy_dens_theta_int(k)) / r;
      p = C0 * pow((r * t), gamm) - hy_pressure_int(k);

      //Compute the flux vector with hyperviscosity
      out_flux(ID_DENS,k,i) = r * w - hv_coef * d3_vals[ID_DENS];
      out_flux(ID_UMOM,k,i) = r * w * u - hv_coef * d3_vals[ID_UMOM];
      out_flux(ID_WMOM,k,i) = r * w * w + p - hv_coef * d3_vals[ID_WMOM];
      out_flux(ID_RHOT,k,i) = r * w * t - hv_coef * d3_vals[ID_RHOT];
    }
  }

  // Use the fluxes to compute tendencies for each cell
  auto in_flux = flux.constMDSpan();
  auto out_tend = tend.mdspan();
  for (int ll = 0; ll < NUM_VARS; ll++){
    for (int k = 0; k < nz; k++){
      for (int i = 0; i < nx; i++){
        Real t = -(in_flux(ll,k+1,i) - in_flux(ll,k,i)) / dz;
        if (ll == ID_WMOM)
          t  = t - state(ID_DENS,k+hs,i+hs) * grav;
        out_tend(ll,k,i) = t;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MiniWeatherArraySequential::
set_halo_values_x(NumArray<double,MDDim3>& nstate)
{
  auto state_in = nstate.constMDSpan();
  auto state_out = nstate.mdspan();

  for (int ll = 0; ll < NUM_VARS; ll++){
    for ( int k = 0; k < nz; k++){
      state_out(ll,k+hs,0) = state_in(ll,k+hs,nx+hs-2);
      state_out(ll,k+hs,1) = state_in(ll,k+hs,nx+hs-1);
      state_out(ll,k+hs,nx+hs) = state_in(ll,k+hs,hs);
      state_out(ll,k+hs,nx+hs+1) = state_in(ll,k+hs,hs+1);
    }
  }

  if (myrank == 0){
    for (int k = 0; k < nz; k++){
      for (int i = 0; i < hs; i++){
        double z = (k_beg + k + 0.5) * dz;
        if (abs(z - 3 * zlen / 4) <= zlen / 16){
          state_out(ID_UMOM,k+hs,i) = (state_in(ID_DENS,k+hs,i) + hy_dens_cell(k + hs)) * 50.;
          state_out(ID_RHOT,k+hs,i) = (state_in(ID_DENS,k+hs,i) + hy_dens_cell(k + hs)) * 298. - hy_dens_theta_cell(k + hs);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Set this task's halo values in the z-direction.
//decomposition in the vertical direction.
void MiniWeatherArraySequential::
set_halo_values_z(NumArray<double,MDDim3>& nstate)
{
  auto state_in = nstate.constMDSpan();
  auto state_out = nstate.mdspan();
  for (int ll = 0; ll < NUM_VARS; ll++){
    for (int i = 0; i < nx + 2 * hs; i++){
      if (ll == ID_WMOM){
        state_out(ll,0,i) = 0.0;
        state_out(ll,1,i) = 0.0;
        state_out(ll,nz+hs,i) = 0.0;
        state_out(ll,nz+hs+1,i) = 0.0;
      }
      else {
        state_out(ll,0,i) = state_in(ll,hs,i);
        state_out(ll,1,i) = state_in(ll,hs,i); // GG: bizarre que ce soit pareil que au dessus.
        state_out(ll,nz+hs,i) = state_in(ll,nz+hs-1,i);
        state_out(ll,nz+hs+1,i) = state_in(ll,nz+hs-1,i); // Idem
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MiniWeatherArraySequential::
init()
{
  int i, k, ii, kk, ll, i_end;
  double x, z, r, u, w, t, hr, ht, nper;

  nranks = 1;
  myrank = 0;

  // For simpler version, replace i_beg = 0, nx = nx_glob, left_rank = 0, right_rank = 0;

  nper = ((double)nx_glob) / nranks;
  i_beg = (int)(round(nper * (myrank)));
  i_end = (int)(round(nper * ((myrank) + 1))) - 1;
  nx = i_end - i_beg + 1;
  left_rank = myrank - 1;
  if (left_rank == -1)
    left_rank = nranks - 1;
  right_rank = myrank + 1;
  if (right_rank == nranks)
    right_rank = 0;

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  k_beg = 0;
  nz = nz_glob;

  // Allocate the model data
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
  dt = dmin(dx, dz) / max_speed * cfl;
  //Set initial elapsed model time and output_counter to zero
  etime = 0.;
  output_counter = 0.;

  // Display grid information

  info() << "nx_glob, nz_glob: " << nx_glob << " " << nz_glob;
  info() << "dx,dz: " << dx << " " << dz;
  info() << "dt: " << dt;

  auto state = nstate.mdspan();

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  for (k = 0; k < nz + 2 * hs; k++) {
    for (i = 0; i < nx + 2 * hs; i++) {
      // Initialize the state to zero
      for (ll = 0; ll < NUM_VARS; ll++)
        nstate(ll,k,i) = 0.0;

      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (kk = 0; kk < nqpoints; kk++) {
        for (ii = 0; ii < nqpoints; ii++) {
          // Compute the x,z location within the global domain based on cell and quadrature index
          x = (i_beg + i - hs + 0.5) * dx + (qpoints[ii] - 0.5) * dx;
          z = (k_beg + k - hs + 0.5) * dz + (qpoints[kk] - 0.5) * dz;

          // Set the fluid state based on the user's specification (default is injection in this example)
          injection(x, z, r, u, w, t, hr, ht);

          // Store into the fluid state array
          state(ID_DENS,k,i) += + r * qweights[ii] * qweights[kk];
          state(ID_UMOM,k,i) += + (r + hr) * u * qweights[ii] * qweights[kk];
          state(ID_WMOM,k,i) += + (r + hr) * w * qweights[ii] * qweights[kk];
          state(ID_RHOT,k,i) += + ((r + hr) * (t + ht) - hr * ht) * qweights[ii] * qweights[kk];
        }
      }

      for (ll = 0; ll < NUM_VARS; ll++)
        nstate_tmp(ll,k,i) = nstate(ll,k,i);
    }
  }

  // Compute the hydrostatic background state over vertical cell averages
  auto out_hy_dens_cell = hy_dens_cell.mdspan();
  auto out_hy_dens_theta_cell = hy_dens_theta_cell.mdspan();
  for (k = 0; k < nz + 2 * hs; k++){
    double dens_cell = 0.0;
    double dens_theta_cell = 0.0;
    for (kk = 0; kk < nqpoints; kk++){
      z = (k_beg + k - hs + 0.5) * dz;

      // Set the fluid state based on the user's specification (default is injection in this example)
      injection(0.0, z, r, u, w, t, hr, ht);

      dens_cell += hr * qweights[kk];
      dens_theta_cell += hr * ht * qweights[kk];
    }
    out_hy_dens_cell(k) = dens_cell;
    out_hy_dens_theta_cell(k) = dens_theta_cell;
  }

  auto out_hy_dens_int = hy_dens_int.mdspan();
  auto out_hy_dens_theta_int = hy_dens_theta_int.mdspan();
  auto out_hy_pressure_int = hy_pressure_int.mdspan();
  // Compute the hydrostatic background state at vertical cell interfaces
  for (k = 0; k < nz + 1; k++) {
    z = (k_beg + k) * dz;

    //Set the fluid state based on the user's specification (default is injection in this example)
    injection(0.0, z, r, u, w, t, hr, ht);

    out_hy_dens_int(k) = hr;
    out_hy_dens_theta_int(k) = hr * ht;
    out_hy_pressure_int(k) = C0 * pow((hr * ht), gamm);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// This test case is initially balanced but injects fast, cold air from the left boundary near the model top
// x and z are input coordinates at which to sample
// r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
// hr and ht are output background hydrostatic density and potential temperature at that location
void MiniWeatherArraySequential::
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
void MiniWeatherArraySequential::
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
void MiniWeatherArraySequential::
output(NumArray<double,MDDim3>& state, double etime)
{
  ARCANE_UNUSED(state);
  ARCANE_UNUSED(etime);
  // Ne fait rien car on n'est pas branché avec 'NetCDF'.
}

// Affiche la somme sur les mailles des variables.
// Cela est utile pour la validation
void MiniWeatherArraySequential::
doExit(RealArrayView reduced_values)
{
  int k, i, ll;
  double sum_v[NUM_VARS];

  auto ns = nstate.constMDSpan();

  for (ll = 0; ll < NUM_VARS; ll++)
    sum_v[ll] = 0.0;
  for (k = 0; k < nz; k++){
    for (i = 0; i < nx + 1; i++){
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

class MiniWeatherArraySequentialService
: public BasicService, public IMiniWeatherService
{
 public:
  explicit MiniWeatherArraySequentialService(const ServiceBuildInfo& sbi)
  : BasicService(sbi), m_p(nullptr){}
  ~MiniWeatherArraySequentialService() override
  {
    delete m_p;
  }
 public:
  void init(IAcceleratorMng*,Int32 nb_x,Int32 nb_z,Real final_time,eMemoryRessource,bool) override
  {
    m_p = new MiniWeatherArraySequential(traceMng(),nb_x,nb_z,final_time);
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
  MiniWeatherArraySequential* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MiniWeatherArraySequentialService,
                        ServiceProperty("MiniWeatherArraySequential",ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IMiniWeatherService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MiniWeatherArraySequential

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
