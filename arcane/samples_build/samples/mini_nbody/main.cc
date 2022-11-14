// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* main.cc                                                     (C) 2000-2022 */
/*                                                                           */
/* Main MiniNBody sample.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/launcher/ArcaneLauncher.h>

#include <iostream>
#include <arcane/utils/NumArray.h>
#include <arcane/utils/ValueConvert.h>

#include <arcane/accelerator/core/IAcceleratorMng.h>

#include <arcane/accelerator/NumArrayViews.h>
#include <arcane/accelerator/RunQueue.h>
#include <arcane/accelerator/RunCommandLoop.h>

#include <chrono>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NBody
{
 public:
  using RealType = float;

  NBody(IAcceleratorMng* acc_mng,Int32 nb_body)
  : m_accelerator_mng(acc_mng), m_nb_body(nb_body)
  {
  }

  void initWithRandom(NumArray<RealType,MDDim1>& data);
  void computeForces();
  void execute();

 private:

  IAcceleratorMng* m_accelerator_mng;
  Int32 m_nb_body;
  RealType m_dt = 0.01f;
  NumArray<RealType,MDDim1> m_x;
  NumArray<RealType,MDDim1> m_y;
  NumArray<RealType,MDDim1> m_z;
  NumArray<RealType,MDDim1> m_vx;
  NumArray<RealType,MDDim1> m_vy;
  NumArray<RealType,MDDim1> m_vz;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NBody::
initWithRandom(NumArray<RealType,MDDim1>& data)
{
  data.resize(m_nb_body);
  for (Int32 i = 0, n = m_nb_body; i < n; i++) {
    data.s(i) = 2.0f * (rand() / (RealType)RAND_MAX) - 1.0f;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NBody::
computeForces()
{
  const Int32 n = m_nb_body;
  const RealType SOFTENING = 1.0e-9f;
  const RealType dt = m_dt;

  {
    // Noyau de calcul déporté sur accélérateur.
    auto command = makeCommand(m_accelerator_mng->defaultQueue());

    auto x = viewIn(command,m_x);
    auto y = viewIn(command,m_y);
    auto z = viewIn(command,m_z);
    auto vx = viewOut(command,m_vx);
    auto vy = viewOut(command,m_vy);
    auto vz = viewOut(command,m_vz);

    const RealType unit = (RealType)1.0;
    const RealType zero = (RealType)0.0;
    command << RUNCOMMAND_LOOP1(iter,m_nb_body) {
      auto [i] = iter();
      RealType fx = zero;
      RealType fy = zero;
      RealType fz = zero;

      for (int j = 0; j < n; j++) {
        RealType dx = x(j) - x(i);
        RealType dy = y(j) - y(i);
        RealType dz = z(j) - z(i);
        RealType inverse_distance = unit / math::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
        RealType d3 = inverse_distance * inverse_distance * inverse_distance;

        fx += dx * d3;
        fy += dy * d3;
        fz += dz * d3;
      }

      vx(i) += dt * fx;
      vy(i) += dt * fy;
      vz(i) += dt * fz;
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NBody::
execute()
{
  const int nb_iter = 10;
  const RealType dt = m_dt;

  std::cout << "NBody::execute nb_body=" << m_nb_body << " nb_iter=" << nb_iter << "\n";

  // Initialisation des positions et vitesses
  initWithRandom(m_x);
  initWithRandom(m_y);
  initWithRandom(m_z);
  initWithRandom(m_vx);
  initWithRandom(m_vy);
  initWithRandom(m_vz);

  const Int32 nb_body = m_nb_body;
  double total_elapsed_time = 0.0;

  for (int iter = 1; iter <= nb_iter; ++iter ) {

    // Calcul les forces
    auto start = std::chrono::high_resolution_clock::now();
    computeForces();
    auto end = std::chrono::high_resolution_clock::now();

    // Ne prend pas en compte la première itération car son temps
    // est souvent non représentatif
    if (iter>1){
      std::chrono::duration<double> duration = (end-start);
      total_elapsed_time += duration.count();
    }

    // Mise à jour des positions
    for (int i = 0 ; i < nb_body; ++i ) {
      m_x.s(i) += m_vx(i)*dt;
      m_y.s(i) += m_vy(i)*dt;
      m_z.s(i) += m_vz(i)*dt;
    }
  }
  std::cout << "Elapsed time_per_iter: " << (total_elapsed_time/(nb_iter-1)) << "s\n";      
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int
main(int argc,char* argv[])
{
  CommandLineArguments args(&argc,&argv);
  ArcaneLauncher::init(args);
    
  // Déclare la fonction qui sera exécutée par l'appel à run()
  auto f = [=](DirectExecutionContext& ctx) -> int
  {
    String nb_body_arg = args.getParameter("NbBody");
    Int32 nb_body = 1000;
    if (!nb_body_arg.null())
      if (builtInGetValue(nb_body,nb_body_arg))
        ARCANE_FATAL("Can not convert '{0}' to integer",nb_body_arg);

    StandaloneAcceleratorMng launcher(ArcaneLauncher::createStandaloneAcceleratorMng());
    IAcceleratorMng* acc_mng = launcher.acceleratorMng();
    NBody nbody(acc_mng,nb_body);
    nbody.execute();
    return 0;
  };
  // Exécute le fonctor 'f'.
  return ArcaneLauncher::run(f);
}
